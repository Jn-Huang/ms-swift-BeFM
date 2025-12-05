#!/bin/bash
#SBATCH --account=qmei
#SBATCH --partition=qmei-a100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --job-name=megatron_qwen3_30b_moe_lora
#SBATCH --output=/home/huangjin/ms-swift-BeFM/logs/qwen3_30b_moe_lora_%j.log
#SBATCH --mail-user=huangjin@umich.edu
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Qwen3-30B-A3B MoE LoRA Training with Megatron-SWIFT
# =============================================================================
# Model: Qwen/Qwen3-30B-A3B-Instruct-2507 (MoE with 128 experts, 8 active)
# Method: LoRA fine-tuning with Megatron parallelism (lower memory)
# Dataset: GSM8K (math reasoning)
# GPUs: 4x A100 80GB (can also work with 2x A100)
# =============================================================================

# Project directory
PROJECT_DIR=/home/huangjin/ms-swift-BeFM

# Important: Load modules for CUDA support
module load cuda/12.8.1
module load python/3.11.5

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

# NCCL settings
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Number of GPUs per node
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

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
echo "=========================================="

# =============================================================================
# Training Configuration
# =============================================================================

# Output directory for checkpoints
OUTPUT_DIR=/scratch/qmei_root/qmei/huangjin/models/megatron_output/Qwen3-30B-A3B-gsm8k-lora

echo "Starting Qwen3-30B-A3B MoE LoRA training with Megatron..."
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# =============================================================================
# Training Command (LoRA)
# =============================================================================
# LoRA training uses less memory than full-parameter training
# With 4 GPUs: PP=2, EP=2, DP=1
# =============================================================================

uv run megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --use_hf true \
    --load_safetensors true \
    --dataset 'hf::gsm8k:main' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    \
    `# === LoRA Settings ===` \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    \
    `# === Parallelism Settings (4 GPUs) ===` \
    `# PP=2: Split model across 2 pipeline stages` \
    `# EP=2: Distribute experts across 2 groups` \
    `# DP=1: Data parallelism = 4 / (2 * 2) = 1` \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --sequence_parallel true \
    \
    `# === MoE Optimization Settings ===` \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --moe_router_dtype fp32 \
    \
    `# === Batch Size Settings ===` \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --packing true \
    \
    `# === Memory Optimization ===` \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    \
    `# === Training Hyperparameters ===` \
    --max_epochs 3 \
    --finetune true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
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
    --merge_lora true \
    \
    `# === Data Loading ===` \
    --num_workers 4 \
    --dataset_num_proc 4 \
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
