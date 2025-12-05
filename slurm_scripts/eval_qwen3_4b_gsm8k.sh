#!/bin/bash
#SBATCH --account=qmei
#SBATCH --partition=qmei-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_qwen3_4b_gsm8k
#SBATCH --output=/home/huangjin/ms-swift-BeFM/logs/eval_qwen3_4b_gsm8k_%j.log
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

# Disable API timeout to prevent timeout errors
# https://swift.readthedocs.io/en/latest/Instruction/Frequently-asked-questions.html
export SWIFT_TIMEOUT=-1

# Verify environment
echo "Python version:"
uv run python --version
echo "PyTorch version:"
uv run python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
uv run python -c "import torch; print(torch.cuda.is_available())"
echo "vLLM version:"
uv run python -c "import vllm; print(vllm.__version__)"
echo "evalscope version:"
uv run python -c "import evalscope; print(evalscope.__version__)"
echo "=========================================="

# Path to fine-tuned LoRA checkpoint
CHECKPOINT_PATH=/scratch/qmei_root/qmei/huangjin/models/ms-swift/qwen3-4b-lora/v3-20251127-165240/checkpoint-1401

# Run evaluation on GSM8K
echo "Starting Qwen3-4B LoRA evaluation on GSM8K with vLLM..."

CUDA_VISIBLE_DEVICES=0 \
uv run swift eval \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf true \
    --adapters $CHECKPOINT_PATH \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k \
    --eval_num_proc 4 \
    --port 8000

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "Job finished at: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Evaluation failed with exit code: $?"
    echo "Job finished at: $(date)"
    echo "=========================================="
    exit 1
fi
