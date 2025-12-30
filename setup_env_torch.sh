#!/bin/bash

# Environment name
ENV_NAME="evogo-torch"
PYTHON_VERSION="3.11"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting environment setup for $ENV_NAME...${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in your PATH."
    exit 1
fi

# Create Conda environment
echo -e "${GREEN}Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION...${NC}"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
# This is a bit tricky in a script. We need to source conda.sh or use conda run.
# For this script to be "one-click" setup, we will use 'conda run' for pip install,
# or ask the user to activate it manually afterwards.
# Better approach: use 'eval "$(conda shell.bash hook)"' to enable activation in script.
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate environment."
    exit 1
fi

# Install swig via conda (required for Box2D)
echo -e "${GREEN}Installing swig...${NC}"
conda install -y swig

# Install dependencies
echo -e "${GREEN}Installing dependencies from requirements_torch.txt...${NC}"
if [ -f "requirements_torch.txt" ]; then
    pip install -r requirements_torch.txt
else
    echo "Error: requirements_torch.txt not found!"
    exit 1
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}To activate the environment, run:${NC}"
echo -e "conda activate $ENV_NAME"

conda activate $ENV_NAME