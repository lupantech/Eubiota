#!/bin/bash

# Science Agent Setup Script
# Supports both inference and training modes

set -e

# Parse command line arguments
MODE="${1:-infer}"  # Default to infer mode if no argument provided

if [[ "$MODE" != "infer" && "$MODE" != "train" ]]; then
    echo "Error: Invalid mode. Use 'infer' or 'train'"
    echo "Usage: bash setup.sh [infer|train]"
    exit 1
fi

echo "======================================"
echo "Science Agent - ${MODE^} Setup"
echo "======================================"

# Install UV if not already installed
if ! command -v uv &> /dev/null
then
    echo "Installing UV package manager..."
    pip install uv
else
    echo "✓ UV is already installed: $(uv --version)"
fi

# Create virtual environment
if [ ! -d ".venv" ]
then
    echo "Creating virtual environment with Python 3.11..."
    uv venv -p 3.11
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
source .venv/bin/activate

# Mode-specific installation
if [ "$MODE" == "infer" ]; then
    # ============== INFERENCE MODE ==============
    echo ""
    echo "Installing Science Agent for inference..."
    echo "Running: uv pip install -e \".[infer]\""
    echo ""
    uv pip install -e ".[infer]"

    echo ""
    echo "======================================"
    echo "✓ Installation Complete!"
    echo "======================================"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the environment:"
    echo "     source .venv/bin/activate"
    echo ""
    echo "  2. Configure your API keys:"
    echo "     cp scientist/.env.template .env"
    echo "     # Edit .env with your API keys"
    echo ""
    echo "  3. Test the installation:"
    echo "     python -c 'from scientist import Agent; print(\"✓ Scientist ready!\")'"
    echo ""
    echo "Optional installations:"
    echo "  • Extended engines: uv pip install -e \".[extended-engines]\""
    echo "  • Development tools: uv pip install -e \".[dev]\""
    echo "  • Everything:       uv pip install -e \".[all]\""
    echo ""

else
    # ============== TRAINING MODE ==============
    echo ""
    echo "Installing Science Agent for training..."
    echo ""

    # Install main project dependencies with training extras first (includes agentops, etc.)
    echo "Installing project dependencies with training extras..."
    uv pip install -e ".[train]"

    # Install trainer package (agentops must be installed first)
    echo "Installing trainer package..."
    cd trainer
    uv pip install -r requirements.txt
    uv pip install --no-deps -e .
    cd ..

    # Install additional dependencies
    echo "Installing additional dependencies..."
    uv pip install omegaconf
    uv pip install codetiming
    uv pip install pyvers multiprocess
    uv pip install dashscope
    uv pip install fire

    # Install AutoGen
    echo "Installing AutoGen..."
    uv pip install "autogen-agentchat" "autogen-ext[openai]"

    # Install LiteLLM
    echo "Installing LiteLLM..."
    uv pip install "litellm[proxy]"

    # Install MCP
    echo "Installing MCP..."
    uv pip install mcp

    # Install OpenAI Agents
    echo "Installing OpenAI Agents..."
    uv pip install openai-agents

    # Install LangChain related packages
    echo "Installing LangChain related packages..."
    uv pip install langgraph "langchain[openai]" langchain-community langchain-text-splitters

    # Install SQL related dependencies
    echo "Installing SQL related dependencies..."
    uv pip install sqlparse nltk

    # Setup stable GPU environment
    echo "Setting up stable GPU environment..."
    bash util/setup_stable_gpu.sh

    # Restart Ray service
    echo "Restarting Ray service..."
    bash util/restart_ray.sh

    echo "Ray server is reflushed."

    # Install system utilities
    echo "Installing system utilities..."
    sudo apt-get update
    sudo apt-get install -y jq
    uv pip install yq

    echo ""
    echo "======================================"
    echo "✓ Training Setup Complete!"
    echo "======================================"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the environment:"
    echo "     source .venv/bin/activate"
    echo ""
    echo "  2. Configure your API keys:"
    echo "     cp scientist/.env.template .env"
    echo "     # Edit .env with your API keys"
    echo ""
    echo "  3. Verify Ray cluster:"
    echo "     ray status"
    echo ""
fi
