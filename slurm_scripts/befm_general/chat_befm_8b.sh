#!/bin/bash
#SBATCH --account=qmei
#SBATCH --partition=qmei-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --job-name=chat_befm_8b
#SBATCH --output=/home/huangjin/ms-swift-BeFM/logs/chat_befm_8b_%j.log
#SBATCH --mail-user=huangjin@umich.edu
#SBATCH --mail-type=END,FAIL

# Project directory
PROJECT_DIR=/home/huangjin/ms-swift-BeFM

# Load modules for CUDA support
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

# Verify environment
echo "Python version:"
uv run python --version
echo "PyTorch version:"
uv run python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
uv run python -c "import torch; print(torch.cuda.is_available())"
echo "=========================================="

# Run interactive chat with Be.FM-8B
echo "Starting chat with befm/Be.FM-8B..."
echo "Type your messages and press Enter. Type 'exit' or Ctrl+C to quit."
echo "=========================================="

uv run swift infer \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --adapters befm/Be.FM-8B \
    --use_hf true \
    --torch_dtype bfloat16 \
    --max_new_tokens 2048 \
    --stream true

echo "=========================================="
echo "Chat session ended at: $(date)"
echo "=========================================="
