#!/bin/bash
# Minimal environment setup script for molax
# Usage: source env.sh

set -e

echo "🚀 Setting up molax development environment..."

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🔨 Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
echo "✨ Activating virtual environment..."
source .venv/bin/activate

# Install package with dev dependencies
echo "📚 Installing molax with dev dependencies..."
uv pip install -e .[dev]

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate this environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
