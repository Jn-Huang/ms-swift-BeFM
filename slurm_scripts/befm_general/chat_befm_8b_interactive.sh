#!/bin/bash
# Interactive chat script - run with: bash slurm_scripts/chat_befm_8b_interactive.sh
# This uses srun to get an interactive GPU session

PROJECT_DIR=/home/huangjin/ms-swift-BeFM

# Request interactive GPU session and run chat
srun --account=qmei \
     --partition=qmei-a100 \
     --gres=gpu:1 \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --mem=48G \
     --time=4:00:00 \
     --pty bash -c "
        module load cuda/12.8.1
        module load python/3.11.5

        cd $PROJECT_DIR

        export HF_HOME=/scratch/qmei_root/qmei/huangjin/.cache/huggingface/
        export TRANSFORMERS_CACHE=/scratch/qmei_root/qmei/huangjin/.cache/huggingface/

        echo '=========================================='
        echo 'GPU Information:'
        nvidia-smi
        echo '=========================================='
        echo 'Starting chat with befm/Be.FM-8B...'
        echo 'Type your messages and press Enter.'
        echo '=========================================='

        uv run swift infer \
            --model meta-llama/Llama-3.1-8B-Instruct \
            --model_type llama3_1 \
            --adapters befm/Be.FM-8B \
            --use_hf true \
            --torch_dtype bfloat16 \
            --max_new_tokens 2048 \
            --stream true
     "
