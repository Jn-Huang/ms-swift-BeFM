#!/bin/bash
#SBATCH --account=qmei
#SBATCH --partition=qmei-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=setup_megatron
#SBATCH --output=/home/huangjin/ms-swift-BeFM/logs/setup_megatron_%j.log
#SBATCH --mail-user=huangjin@umich.edu
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Setup Megatron-SWIFT environment using uv (requires GPU for compilation)
# =============================================================================

PROJECT_DIR=/home/huangjin/ms-swift-BeFM
cd $PROJECT_DIR

echo "=========================================="
echo "Setting up Megatron-SWIFT environment with uv"
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# Load required modules
module load cuda/12.8.1
module load cudnn/12.8-v9.10.0
module load python/3.11.5

# Verify GPU access
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# =============================================================================
# Set environment variables for compilation
# =============================================================================

# cuDNN paths
export CUDNN_PATH=$CUDNN_ROOT
export CUDNN_INCLUDE_DIR=$CUDNN_ROOT/include
export CUDNN_LIBRARY=$CUDNN_ROOT/lib64

# NCCL paths (from PyTorch's nvidia package - will be available after base install)
# We'll set this after initial sync

# Compiler include paths
export CPLUS_INCLUDE_PATH=$CUDNN_ROOT/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CUDNN_ROOT/include:$C_INCLUDE_PATH
export CPATH=$CUDNN_ROOT/include:$CPATH
export LIBRARY_PATH=$CUDNN_ROOT/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_ROOT/lib64:$LD_LIBRARY_PATH

# HuggingFace cache
export HF_HOME=/scratch/qmei_root/qmei/huangjin/.cache/huggingface/
export TRANSFORMERS_CACHE=/scratch/qmei_root/qmei/huangjin/.cache/huggingface/

echo "Environment variables set:"
echo "  CUDNN_PATH: $CUDNN_PATH"
echo "  CPLUS_INCLUDE_PATH: $CPLUS_INCLUDE_PATH"

# =============================================================================
# Step 1: Sync base dependencies with uv
# =============================================================================

echo "=========================================="
echo "Step 1: Syncing base dependencies with uv..."
echo "=========================================="

uv sync

# =============================================================================
# Step 2: Add NCCL paths (now available from nvidia-nccl-cu12 package)
# =============================================================================

echo "=========================================="
echo "Step 2: Setting up NCCL paths..."
echo "=========================================="

# Find NCCL in the venv
NCCL_PATH=$(uv run python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])" 2>/dev/null)
if [ -n "$NCCL_PATH" ]; then
    export NCCL_INCLUDE_DIR=$NCCL_PATH/include
    export NCCL_LIB_DIR=$NCCL_PATH/lib
    export CPLUS_INCLUDE_PATH=$NCCL_INCLUDE_DIR:$CPLUS_INCLUDE_PATH
    export C_INCLUDE_PATH=$NCCL_INCLUDE_DIR:$C_INCLUDE_PATH
    export CPATH=$NCCL_INCLUDE_DIR:$CPATH
    export LIBRARY_PATH=$NCCL_LIB_DIR:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$NCCL_LIB_DIR:$LD_LIBRARY_PATH
    echo "NCCL found at: $NCCL_PATH"
else
    echo "WARNING: NCCL not found in venv"
fi

# =============================================================================
# Step 3: Install Megatron dependencies with uv
# =============================================================================

echo "=========================================="
echo "Step 3: Installing Megatron dependencies (megatron-core)..."
echo "=========================================="

uv sync --extra megatron

echo "=========================================="
echo "Step 3b: Installing transformer_engine (requires special handling)..."
echo "=========================================="

# transformer_engine needs to be installed from git source
# The PyPI version has build issues with build_tools module
# Use --python to ensure we install into the project venv
uv pip install --python $PROJECT_DIR/.venv/bin/python --no-build-isolation \
    "git+https://github.com/NVIDIA/TransformerEngine.git@v2.9.0#egg=transformer_engine[pytorch]"

echo "=========================================="
echo "Step 3c: Installing flash-attn..."
echo "=========================================="

MAX_JOBS=8 uv pip install --python $PROJECT_DIR/.venv/bin/python --no-build-isolation "flash-attn<2.8.2"

# =============================================================================
# Step 4: Clone Megatron-LM repo (for training scripts)
# =============================================================================

echo "=========================================="
echo "Step 4: Cloning Megatron-LM repository..."
echo "=========================================="

MEGATRON_PATH=/scratch/qmei_root/qmei/huangjin/Megatron-LM
if [ ! -d "$MEGATRON_PATH" ]; then
    git clone --branch core_r0.14.0 https://github.com/NVIDIA/Megatron-LM.git $MEGATRON_PATH
else
    echo "Megatron-LM already exists at $MEGATRON_PATH"
fi

# =============================================================================
# Step 5: Verify Installation
# =============================================================================

echo "=========================================="
echo "Step 5: Verifying installation..."
echo "=========================================="

uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Number of GPUs: {torch.cuda.device_count()}')

try:
    import transformer_engine
    print(f'TransformerEngine: OK')
except ImportError as e:
    print(f'TransformerEngine: FAILED - {e}')

try:
    import megatron.core
    print(f'Megatron-Core: OK')
except ImportError as e:
    print(f'Megatron-Core: FAILED - {e}')

try:
    import flash_attn
    print(f'Flash-Attention: OK ({flash_attn.__version__})')
except ImportError as e:
    print(f'Flash-Attention: FAILED - {e}')
"

echo "=========================================="
echo "Environment setup complete!"
echo "Job finished at: $(date)"
echo "=========================================="
