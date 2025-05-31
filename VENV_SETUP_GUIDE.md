# Virtual Environment Setup Guide for Chatterbox TTS

This guide provides step-by-step instructions to properly set up, activate, and install all dependencies for the Chatterbox TTS project using a Python virtual environment.

## Prerequisites

- Python 3.8 or higher installed on your system
- Git (for cloning the repository)
- CUDA-compatible GPU (optional but recommended for better performance)

## Step-by-Step Setup Instructions

### 1. Clone the Repository (if not already done)

```bash
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
```

### 2. Create Virtual Environment

Choose the method based on your Python installation:

#### Option A: Using `venv` (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Alternative for some systems
python3 -m venv venv
```

#### Option B: Using `virtualenv` (if installed)
```bash
virtualenv venv
```

### 3. Activate Virtual Environment

#### On Windows (PowerShell/Command Prompt)
```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# Command Prompt
venv\Scripts\activate.bat
```

#### On macOS/Linux
```bash
source venv/bin/activate
```

**Note:** After activation, you should see `(venv)` at the beginning of your command prompt, indicating the virtual environment is active.

### 4. Upgrade pip (Recommended)

```bash
python -m pip install --upgrade pip
```

### 5. Install Dependencies

#### Method 1: Install from PyPI (Recommended for users)
```bash
pip install chatterbox-tts
```

#### Method 2: Install from source (For development)
```bash
# Install with constraints to avoid dependency conflicts
pip install -c constraints.txt -e .

# Or install dependencies manually
pip install -c constraints.txt \
    "numpy~=1.26.0" \
    "resampy==0.4.3" \
    "librosa==0.11.0" \
    "s3tokenizer" \
    "torch==2.6.0" \
    "torchaudio==2.6.0" \
    "transformers==4.46.3" \
    "diffusers==0.29.0" \
    "resemble-perth==1.0.1" \
    "omegaconf==2.3.0" \
    "conformer==0.3.2" \
    "safetensors==0.5.3"
```

### 6. Install Additional Dependencies (Optional)

For running the Gradio demo applications:
```bash
pip install gradio
```

For development and testing:
```bash
pip install pytest black flake8
```

### 7. Verify Installation

Test the installation by running the example script:

```bash
python example_tts.py
```

Or test with a simple Python command:
```python
python -c "import chatterbox; print('Chatterbox TTS installed successfully!')"
```

## GPU Setup (Optional but Recommended)

### CUDA Installation
If you have an NVIDIA GPU, ensure CUDA is properly installed:

1. Check CUDA availability:
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

2. If CUDA is not available, install the appropriate PyTorch version:
```bash
# For CUDA 11.8
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Permission Errors (Windows)
If you encounter permission errors when activating the virtual environment:
```powershell
# Run PowerShell as Administrator and execute:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 2. Python Version Conflicts
Ensure you're using Python 3.8 or higher:
```bash
python --version
```

#### 3. Dependency Conflicts
If you encounter dependency conflicts, use the constraints file:
```bash
pip install -c constraints.txt -r requirements.txt
```

#### 4. Out of Memory Errors
For systems with limited RAM, you can try CPU-only installation:
```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 5. Audio Library Issues (Linux)
On some Linux distributions, you might need additional audio libraries:
```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1 ffmpeg

# CentOS/RHEL
sudo yum install libsndfile ffmpeg
```

## Deactivating the Virtual Environment

When you're done working with the project:
```bash
deactivate
```

## Environment Management Tips

### 1. Save Current Environment
To save your current environment setup:
```bash
pip freeze > requirements.txt
```

### 2. Recreate Environment
To recreate the environment on another machine:
```bash
pip install -r requirements.txt
```

### 3. Clean Installation
If you need to start fresh:
```bash
# Deactivate and remove virtual environment
deactivate
rm -rf venv  # On Windows: rmdir /s venv

# Create new environment and follow steps 2-5 again
```

## Quick Start Commands Summary

```bash
# 1. Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

# 2. Upgrade pip and install
python -m pip install --upgrade pip
pip install chatterbox-tts

# 3. Test installation
python example_tts.py
```

## Additional Resources

- [Chatterbox TTS Documentation](https://github.com/resemble-ai/chatterbox)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)

## Support

If you encounter issues:
1. Check the [GitHub Issues](https://github.com/resemble-ai/chatterbox/issues)
2. Join the [Discord Community](https://discord.gg/XqS7RxUp)
3. Review the troubleshooting section above

---

**Note:** Always ensure your virtual environment is activated (you should see `(venv)` in your terminal prompt) before installing packages or running the application. 