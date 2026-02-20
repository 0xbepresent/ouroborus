# Ouroborus Server Setup Guide

This guide explains how to set up Ouroborus on your personal server to run security analysis for **C/C++** (using CodeQL) and **Solidity** (using Slither) smart contracts.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [C/C++ Analysis Setup (CodeQL)](#cc-analysis-setup-codeql)
4. [Solidity Analysis Setup (Slither)](#solidity-analysis-setup-slither)
5. [LLM Configuration](#llm-configuration)
6. [Running Analysis](#running-analysis)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 20GB free space
- **Python**: 3.10 or higher
- **Git**: 2.x

### For C/C++ Analysis (CodeQL)
- **CodeQL CLI**: v2.15.0 or higher
- **GitHub Token**: Required for downloading CodeQL databases

### For Solidity Analysis (Slither)
- **Slither**: 0.10.0 or higher
- **solc** (Solidity compiler): Multiple versions via solc-select
- **Node.js**: 16+ (for Hardhat/Foundry projects)

---

## Quick Start

```bash
# 1. Clone your fork
git clone https://github.com/YOUR_USERNAME/ouroborus.git
cd ouroborus

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: .\venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install poetry
poetry install

# 4. Copy and configure .env
cp .env.example .env
# Edit .env with your API keys

# 5. Run analysis
# For C/C++:
ouroborus redis/redis --lang c

# For Solidity:
ouroborus OpenZeppelin/openzeppelin-contracts --lang solidity
```

---

## C/C++ Analysis Setup (CodeQL)

### 1. Install CodeQL CLI

#### Option A: Download from GitHub
```bash
# Download latest release
wget https://github.com/github/codeql-cli-binaries/releases/latest/download/codeql-linux64.zip

# Extract
unzip codeql-linux64.zip

# Move to /opt (or another location)
sudo mv codeql /opt/codeql

# Add to PATH
echo 'export PATH="/opt/codeql:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
codeql version
```

#### Option B: Using Homebrew (macOS)
```bash
brew install codeql
```

### 2. Configure CodeQL Path

Add to your `.env` file:
```env
CODEQL_PATH=/opt/codeql/codeql
```

### 3. GitHub Token (Required)

CodeQL databases are fetched from GitHub's API. You need a token with `public_repo` scope.

1. Go to https://github.com/settings/tokens
2. Generate a new token (classic) with `public_repo` scope
3. Add to `.env`:
```env
GITHUB_TOKEN=ghp_your_token_here
```

---

## Solidity Analysis Setup (Slither)

### 1. Install Slither

```bash
# Using pip (recommended)
pip install slither-analyzer

# Or using pipx (isolated)
pipx install slither-analyzer

# Verify installation
slither --version
```

### 2. Install solc-select (Multiple Solidity Versions)

Most Solidity projects require specific compiler versions. `solc-select` lets you install and switch between versions.

```bash
# Install solc-select
pip install solc-select

# Install common Solidity versions
solc-select install 0.8.20
solc-select install 0.8.19
solc-select install 0.8.0
solc-select install 0.7.6
solc-select install 0.6.12

# Set default version
solc-select use 0.8.20

# Verify
solc --version
```

### 3. Install Node.js (for Hardhat/Foundry projects)

Many Solidity projects use Hardhat or Foundry. These require Node.js for dependency resolution.

```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18

# Or using apt (Ubuntu)
sudo apt update
sudo apt install nodejs npm
```

### 4. Install Foundry (Optional but Recommended)

```bash
curl -L https://foundry.paradigm.xyz | bash
source ~/.bashrc
foundryup
```

---

## LLM Configuration

Ouroborus uses LiteLLM to support multiple LLM providers. Configure one in your `.env` file:

### OpenAI
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-your-key-here
```

### Anthropic (Claude)
```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
LLM_API_KEY=sk-ant-your-key-here
```

### Azure OpenAI
```env
LLM_PROVIDER=azure
LLM_MODEL=gpt-4o
LLM_API_KEY=your-azure-key
LLM_ENDPOINT=https://your-resource.openai.azure.com/
LLM_API_VERSION=2024-02-15-preview
```

### Ollama (Local/Self-hosted)
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
LLM_ENDPOINT=http://localhost:11434
```

### Hugging Face
```env
LLM_PROVIDER=huggingface
LLM_MODEL=meta-llama/Llama-3.1-70B-Instruct
LLM_API_KEY=hf_your-key-here
```

---

## Running Analysis

### C/C++ Analysis (CodeQL)

```bash
# Analyze a specific repository
ouroborus redis/redis --lang c

# Force re-download of database
ouroborus redis/redis --lang c --force

# With custom threads
ouroborus redis/redis --lang c
```

### Solidity Analysis (Slither)

```bash
# Analyze OpenZeppelin contracts
ouroborus OpenZeppelin/openzeppelin-contracts --lang solidity

# Analyze Uniswap
ouroborus Uniswap/v3-core --lang solidity

# Force re-clone
ouroborus OpenZeppelin/openzeppelin-contracts --lang solidity --force
```

### View Results

```bash
# Open the UI
ouroborus-ui

# List analyzed repos
ouroborus-list
```

### Validate Configuration

```bash
ouroborus-validate
```

---

## Directory Structure After Analysis

```
output/
├── databases/          # CodeQL databases (C/C++)
│   └── c/
│       └── redis/
├── repos/              # Cloned repositories (Solidity)
│   └── solidity/
│       └── openzeppelin-contracts/
├── slither_results/    # Raw Slither JSON output
├── results/            # LLM analysis results
│   ├── c/
│   │   └── Copy_function_using_source_size/
│   └── solidity/
│       ├── reentrancy-eth/
│       └── unchecked-transfer/
└── zip_dbs/            # Downloaded database ZIPs
```

---

## Troubleshooting

### Slither: "solc not found"

```bash
# Install the required solc version
solc-select install 0.8.20
solc-select use 0.8.20
```

### Slither: "Compilation failed"

Many projects need dependencies installed first:

```bash
cd output/repos/solidity/your-project

# For Hardhat projects
npm install

# For Foundry projects
forge install
```

### CodeQL: "Database not found"

Not all repositories have CodeQL databases available. The repository must:
1. Have CodeQL analysis enabled in GitHub Actions
2. Have a C/C++ codebase
3. Have publicly accessible databases

### LLM: "Rate limit exceeded"

- Reduce the number of issues being analyzed
- Use a different LLM provider
- Wait for rate limits to reset

### Permission denied errors

```bash
# Fix permissions
chmod -R 755 output/
```

---

## Automated Deployment Script

Create a setup script for your server:

```bash
#!/bin/bash
# setup_ouroborus.sh

set -e

echo "🚀 Setting up Ouroborus..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install -y python3.10 python3.10-venv python3-pip

# Install Git
sudo apt install -y git

# Clone Ouroborus
git clone https://github.com/YOUR_USERNAME/ouroborus.git
cd ouroborus

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install poetry
poetry install

# Install Slither for Solidity
pip install slither-analyzer solc-select

# Install common Solidity versions
solc-select install 0.8.20 0.8.19 0.8.0 0.7.6 0.6.12
solc-select use 0.8.20

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install CodeQL (optional, for C/C++)
wget https://github.com/github/codeql-cli-binaries/releases/latest/download/codeql-linux64.zip
unzip codeql-linux64.zip
sudo mv codeql /opt/codeql
echo 'export PATH="/opt/codeql:$PATH"' >> ~/.bashrc

echo "✅ Setup complete!"
echo "Configure your .env file and run: ouroborus <repo> --lang <c|solidity>"
```

---

## Running as a Service (Optional)

Create a systemd service for background analysis:

```ini
# /etc/systemd/system/ouroborus.service
[Unit]
Description=Ouroborus Security Analysis
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/home/your-user/ouroborus
Environment=PATH=/home/your-user/ouroborus/venv/bin:/opt/codeql:$PATH
ExecStart=/home/your-user/ouroborus/venv/bin/python -m src.pipeline your-repo --lang solidity
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## Security Notes

1. **API Keys**: Never commit `.env` files with real API keys
2. **GitHub Token**: Use tokens with minimal required permissions
3. **Network**: Consider running analysis in an isolated network
4. **Output**: Analysis results may contain sensitive code snippets

---

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/YOUR_USERNAME/ouroborus/issues)
- Documentation: See README.md for more details
