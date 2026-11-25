#!/usr/bin/env bash
# Run inside a fresh environment:
#   conda create -n instok3d python=3.10 -y && conda activate instok3d
set -e

pip install --upgrade pip
pip install setuptools==68.0.0 packaging ninja

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install numpy==1.26.4

# CUDA extensions (need nvcc on the PATH)
pip install external/pointops --no-build-isolation
pip install git+https://github.com/nerfstudio-project/gsplat.git@3d4f9027f36d70ac8e89536e14c247de377f74b5 --no-build-isolation

pip install -r requirements.txt

# OpenAI CLIP: text encoder of LSeg, only needed for feature lifting
pip install git+https://github.com/openai/CLIP.git@04f4dc2ca1ed0acc9893bd1a3b526a7e02c4bb10 --no-build-isolation

echo "Setup complete."
