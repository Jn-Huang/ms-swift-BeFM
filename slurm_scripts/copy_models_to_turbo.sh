#!/bin/bash
#SBATCH --account=qmei
#SBATCH --partition=qmei-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --job-name=copy_models_turbo
#SBATCH --output=/home/huangjin/ms-swift-BeFM/logs/copy_models_turbo_%j.log
#SBATCH --mail-user=huangjin@umich.edu
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Source and destination paths
SRC_HF_CACHE=/scratch/qmei_root/qmei/huangjin/.cache/huggingface/hub
DST_HF_CACHE=/nfs/turbo/si-qmei/huangjin/.cache/huggingface/hub

SRC_MODELS=/scratch/qmei_root/qmei/huangjin/models/ms-swift
DST_MODELS=/nfs/turbo/si-qmei/huangjin/models/ms-swift

# Create destination directories
echo "Creating destination directories..."
mkdir -p $DST_HF_CACHE
mkdir -p $DST_MODELS

# Copy Qwen3-4B model
echo "=========================================="
echo "Copying Qwen3-4B-Instruct-2507 (8.5G)..."
echo "Started at: $(date)"
rsync -av --progress $SRC_HF_CACHE/models--Qwen--Qwen3-4B-Instruct-2507 $DST_HF_CACHE/
echo "Completed at: $(date)"

# Copy LLaMA 3.3-70B model
echo "=========================================="
echo "Copying Llama-3.3-70B-Instruct (297G)..."
echo "Started at: $(date)"
rsync -av --progress $SRC_HF_CACHE/models--meta-llama--Llama-3.3-70B-Instruct $DST_HF_CACHE/
echo "Completed at: $(date)"

# Copy Qwen3-4B LoRA
echo "=========================================="
echo "Copying qwen3-4b-lora (430M)..."
echo "Started at: $(date)"
rsync -av --progress $SRC_MODELS/qwen3-4b-lora $DST_MODELS/
echo "Completed at: $(date)"

# Copy LLaMA 70B LoRA
echo "=========================================="
echo "Copying llama3-70b-lora (4.6G)..."
echo "Started at: $(date)"
rsync -av --progress $SRC_MODELS/llama3-70b-lora $DST_MODELS/
echo "Completed at: $(date)"

# Verify copies
echo "=========================================="
echo "Verifying copied files..."
echo "HF Cache contents:"
ls -la $DST_HF_CACHE/
echo ""
echo "LoRA models:"
ls -la $DST_MODELS/

# Show sizes
echo "=========================================="
echo "Copied sizes:"
du -sh $DST_HF_CACHE/models--Qwen--Qwen3-4B-Instruct-2507
du -sh $DST_HF_CACHE/models--meta-llama--Llama-3.3-70B-Instruct
du -sh $DST_MODELS/qwen3-4b-lora
du -sh $DST_MODELS/llama3-70b-lora

echo "=========================================="
echo "All copies completed!"
echo "Job finished at: $(date)"
echo "=========================================="
