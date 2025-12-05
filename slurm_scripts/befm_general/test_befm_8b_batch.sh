#!/bin/bash
#SBATCH --account=qmei
#SBATCH --partition=qmei-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --job-name=test_befm_8b
#SBATCH --output=/home/huangjin/ms-swift-BeFM/logs/befm_general/test_befm_8b_%j.log
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

# Create output directory
mkdir -p $PROJECT_DIR/logs/befm_general

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

# Run batch inference with Be.FM-8B
echo "Starting batch inference with befm/Be.FM-8B..."
echo "Input: $PROJECT_DIR/logs/befm_general/test_questions.jsonl"
echo "=========================================="

uv run swift infer \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model_type llama3_1 \
    --adapters befm/Be.FM-8B \
    --use_hf true \
    --torch_dtype bfloat16 \
    --max_new_tokens 256 \
    --stream false \
    --val_dataset "$PROJECT_DIR/logs/befm_general/test_questions.jsonl" \
    --result_path "$PROJECT_DIR/logs/befm_general/befm_8b_test_results_${SLURM_JOB_ID}.jsonl"

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Batch inference completed successfully!"
    echo "Results saved to: $PROJECT_DIR/logs/befm_general/befm_8b_test_results_${SLURM_JOB_ID}.jsonl"
    echo "Job finished at: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Batch inference failed with exit code: $?"
    echo "Job finished at: $(date)"
    echo "=========================================="
    exit 1
fi
