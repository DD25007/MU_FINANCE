#!/bin/bash
# Setup script for LoRA-Credit-Unlearn
# Creates conda env 'lora_mu' with all required packages

ENV_NAME="lora_mu"

echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.11 -y

echo "Activating environment..."
source activate $ENV_NAME || conda activate $ENV_NAME

echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "Installing remaining packages..."
pip install -r requirements.txt --upgrade

echo ""
echo "Done! Activate with:  conda activate $ENV_NAME"
echo "Run pipeline with:    python main.py --mode quick"
