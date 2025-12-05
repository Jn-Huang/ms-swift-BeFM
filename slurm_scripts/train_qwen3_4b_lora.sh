#!/bin/bash
#SBATCH --account=qmei
#SBATCH --partition=qmei-a100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=swift_qwen3_4b_lora
#SBATCH --output=/home/huangjin/ms-swift-BeFM/logs/qwen3_4b_lora_%j.log
#SBATCH --mail-user=huangjin@umich.edu
#SBATCH --mail-type=END,FAIL

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

# Create logs directory if it doesn't exist
mkdir -p $PROJECT_DIR/logs

# Set HuggingFace cache paths
export HF_HOME=/scratch/qmei_root/qmei/huangjin/.cache/huggingface/
export TRANSFORMERS_CACHE=/scratch/qmei_root/qmei/huangjin/.cache/huggingface/

# Set wandb project name
export WANDB_PROJECT="ms-swift-experiments"

# Verify environment
echo "Python version:"
uv run python --version
echo "PyTorch version:"
uv run python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
uv run python -c "import torch; print(torch.cuda.is_available())"
echo "Number of GPUs:"
uv run python -c "import torch; print(torch.cuda.device_count())"
echo "=========================================="

# Run training with ms-swift
echo "Starting Qwen3-4B LoRA SFT training with 2 GPUs..."

NPROC_PER_NODE=2 \
uv run swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf true \
    --train_type lora \
    --dataset 'hf::gsm8k:main' \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --max_length 2048 \
    --output_dir /scratch/qmei_root/qmei/huangjin/models/ms-swift/qwen3-4b-lora \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name qwen3_4b_lora_sft

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Job finished at: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training failed with exit code: $?"
    echo "Job finished at: $(date)"
    echo "=========================================="
    exit 1
fi
