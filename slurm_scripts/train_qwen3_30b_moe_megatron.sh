#!/bin/bash
#SBATCH --account=qmei
#SBATCH --partition=qmei-a100
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --job-name=megatron_qwen3_30b_moe
#SBATCH --output=/home/huangjin/ms-swift-BeFM/logs/qwen3_30b_moe_megatron_%j.log
#SBATCH --mail-user=huangjin@umich.edu
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Qwen3-30B-A3B MoE Training with Megatron-SWIFT
# =============================================================================
# Model: Qwen/Qwen3-30B-A3B-Instruct-2507 (MoE with 128 experts, 8 active)
# Method: Full-parameter SFT with Megatron parallelism
# Dataset: GSM8K (math reasoning)
# GPUs: 8x A100 80GB
# =============================================================================

# Project directory
PROJECT_DIR=/home/huangjin/ms-swift-BeFM

# Important: Load modules for CUDA support
module load cuda/12.8.1
module load cudnn/12.8-v9.10.0
module load python/3.11.5

# Use miniconda environment where Megatron dependencies are installed
source /home/huangjin/miniconda3/etc/profile.d/conda.sh
conda activate base

# Set NCCL paths for runtime
export LD_LIBRARY_PATH=/home/huangjin/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Display GPU information
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Change to project directory
cd $PROJECT_DIR || exit 1

# Create directories
mkdir -p $PROJECT_DIR/logs
mkdir -p /scratch/qmei_root/qmei/huangjin/models/megatron_output

# =============================================================================
# Environment Variables
# =============================================================================

# HuggingFace cache paths
export HF_HOME=/scratch/qmei_root/qmei/huangjin/.cache/huggingface/
export TRANSFORMERS_CACHE=/scratch/qmei_root/qmei/huangjin/.cache/huggingface/
export HF_DATASETS_CACHE=/scratch/qmei_root/qmei/huangjin/.cache/huggingface/datasets

# Megatron-LM path (if pre-cloned)
export MEGATRON_LM_PATH=/scratch/qmei_root/qmei/huangjin/Megatron-LM

# WandB settings
export WANDB_PROJECT="ms-swift-megatron"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# NCCL settings for better multi-GPU communication
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Number of GPUs per node
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# =============================================================================
# Verify Environment
# =============================================================================

echo "Python version:"
uv run python --version
echo "PyTorch version:"
uv run python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
uv run python -c "import torch; print(torch.cuda.is_available())"
echo "Number of GPUs:"
uv run python -c "import torch; print(torch.cuda.device_count())"
echo "TransformerEngine:"
uv run python -c "import transformer_engine; print('OK')" 2>/dev/null || echo "Not installed"
echo "Megatron-Core:"
uv run python -c "import megatron.core; print('OK')" 2>/dev/null || echo "Not installed"
echo "=========================================="

# =============================================================================
# Training Configuration
# =============================================================================

# Output directory for checkpoints
OUTPUT_DIR=/scratch/qmei_root/qmei/huangjin/models/megatron_output/Qwen3-30B-A3B-gsm8k

echo "Starting Qwen3-30B-A3B MoE SFT training with Megatron..."
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# =============================================================================
# Training Command
# =============================================================================
# Using mcore-bridge mode (--load_safetensors) to avoid weight conversion step
# This loads HuggingFace safetensors directly without pre-conversion
# =============================================================================

uv run megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --use_hf true \
    --load_safetensors true \
    --dataset 'hf::gsm8k:main' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    \
    `# === Parallelism Settings ===` \
    `# PP=2: Split model across 2 pipeline stages` \
    `# EP=4: Distribute 128 experts across 4 groups (32 experts per group)` \
    `# DP=1: Data parallelism = 8 GPUs / (PP=2 * EP=4) = 1` \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --sequence_parallel true \
    \
    `# === MoE Optimization Settings ===` \
    `# These are critical for MoE training performance` \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --moe_router_dtype fp32 \
    \
    `# === Batch Size Settings ===` \
    `# micro_batch_size: Per-GPU batch size per forward pass` \
    `# global_batch_size: Total batch size across all GPUs` \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --packing true \
    \
    `# === Memory Optimization (Activation Checkpointing) ===` \
    `# Recompute activations to save memory at cost of compute` \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    \
    `# === Training Hyperparameters ===` \
    --max_epochs 3 \
    --finetune true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --weight_decay 0.1 \
    --clip_grad 1.0 \
    \
    `# === Sequence Length ===` \
    --max_length 2048 \
    \
    `# === Performance Optimizations ===` \
    --cross_entropy_loss_fusion true \
    --attention_backend flash \
    --overlap_grad_reduce true \
    --overlap_param_gather true \
    \
    `# === Checkpoint Settings ===` \
    --save $OUTPUT_DIR \
    --save_interval 200 \
    --eval_interval 200 \
    --no_save_optim true \
    --no_save_rng true \
    --save_safetensors true \
    \
    `# === Data Loading ===` \
    --num_workers 8 \
    --dataset_num_proc 8 \
    \
    `# === Logging ===` \
    --log_interval 5

# =============================================================================
# Check Exit Status
# =============================================================================

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Checkpoints saved to: $OUTPUT_DIR"
    echo "Job finished at: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training failed with exit code: $?"
    echo "Job finished at: $(date)"
    echo "=========================================="
    exit 1
fi
