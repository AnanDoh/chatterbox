# Flux + Chatterbox Integration Guide

## Overview
This guide explains how to install and use Flux for image generation alongside your existing Chatterbox TTS setup without breaking either installation. Both systems will coexist using separate virtual environments and shared PyTorch installations where compatible.

## Current Chatterbox Setup Analysis
- **Python Version**: 3.8+ (as per pyproject.toml)
- **PyTorch Version**: 2.6.0 (compatible with Flux)
- **Virtual Environment**: `venv/` directory
- **Key Dependencies**: transformers, diffusers, torch, torchaudio
- **GPU Support**: CUDA-enabled setup

## Strategy: Separate Virtual Environments with Shared Base

### Why This Approach?
1. **Isolation**: Prevents dependency conflicts between Chatterbox and Flux
2. **Compatibility**: Both use PyTorch 2.6.0, which is compatible
3. **Flexibility**: Can update either system independently
4. **Safety**: Preserves your working Chatterbox installation

## Installation Steps

### Step 1: Create Flux Virtual Environment

```powershell
# Create a separate virtual environment for Flux
python -m venv flux_venv

# Activate the Flux environment
.\flux_venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 2: Install Flux Dependencies

```powershell
# Install PyTorch 2.6.0 (same version as Chatterbox for compatibility)
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# Install Flux-specific dependencies
pip install diffusers==0.29.0
pip install transformers==4.46.3
pip install accelerate
pip install safetensors==0.5.3
pip install Pillow
pip install numpy~=1.26.0

# For GGUF quantized models (memory efficient)
pip install gguf

# Optional: For ComfyUI integration
pip install comfyui
```

### Step 3: Install mflux (Mac-optimized Flux, works on Windows too)

```powershell
# Alternative: Install mflux for easier Flux usage
pip install mflux
```

### Step 4: Create Flux Test Script

Create `test_flux.py`:

```python
import torch
from diffusers import FluxPipeline

def test_flux_installation():
    """Test Flux installation and generate a sample image."""
    
    print("Testing Flux installation...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Use Flux Schnell (faster, Apache 2.0 license)
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        
        # Generate test image
        prompt = "A beautiful sunset over mountains, digital art"
        image = pipe(
            prompt,
            num_inference_steps=4,  # Schnell works well with 4 steps
            guidance_scale=0.0,     # Schnell doesn't need guidance
            height=512,
            width=512
        ).images[0]
        
        # Save image
        image.save("flux_test_output.png")
        print("✅ Flux test successful! Image saved as 'flux_test_output.png'")
        
    except Exception as e:
        print(f"❌ Flux test failed: {e}")

if __name__ == "__main__":
    test_flux_installation()
```

### Step 5: Create Environment Management Scripts

Create `activate_chatterbox.bat`:
```batch
@echo off
echo Activating Chatterbox environment...
call venv\Scripts\activate.bat
echo Chatterbox environment activated. PyTorch version:
python -c "import torch; print(torch.__version__)"
```

Create `activate_flux.bat`:
```batch
@echo off
echo Activating Flux environment...
call flux_venv\Scripts\activate.bat
echo Flux environment activated. PyTorch version:
python -c "import torch; print(torch.__version__)"
```

### Step 6: Create Combined Usage Script

Create `flux_chatterbox_demo.py`:

```python
"""
Combined Flux + Chatterbox Demo
Demonstrates how to use both systems together
"""

import subprocess
import sys
import os
from pathlib import Path

def run_in_venv(venv_path, script_content, script_name):
    """Run Python code in a specific virtual environment."""
    
    # Create temporary script
    temp_script = Path(script_name)
    temp_script.write_text(script_content)
    
    try:
        # Determine activation script based on OS
        if os.name == 'nt':  # Windows
            activate_script = venv_path / "Scripts" / "activate.bat"
            cmd = f'"{activate_script}" && python {script_name}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:  # Unix-like
            activate_script = venv_path / "bin" / "activate"
            cmd = f'source "{activate_script}" && python {script_name}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
        
        return result.returncode == 0, result.stdout, result.stderr
    
    finally:
        # Clean up temporary script
        if temp_script.exists():
            temp_script.unlink()

def generate_image_with_flux(prompt, output_path="generated_image.png"):
    """Generate image using Flux in its virtual environment."""
    
    flux_script = f'''
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

image = pipe(
    "{prompt}",
    num_inference_steps=4,
    guidance_scale=0.0,
    height=512,
    width=512
).images[0]

image.save("{output_path}")
print("Image generated successfully!")
'''
    
    success, stdout, stderr = run_in_venv(Path("flux_venv"), flux_script, "temp_flux.py")
    
    if success:
        print(f"✅ Image generated: {output_path}")
        return output_path
    else:
        print(f"❌ Image generation failed: {stderr}")
        return None

def generate_speech_with_chatterbox(text, output_path="generated_speech.wav"):
    """Generate speech using Chatterbox in its virtual environment."""
    
    chatterbox_script = f'''
# Add your Chatterbox TTS code here
# This is a placeholder - replace with actual Chatterbox usage
print("Generating speech with Chatterbox...")
print("Text: {text}")
print("Output: {output_path}")
# Your existing Chatterbox code would go here
'''
    
    success, stdout, stderr = run_in_venv(Path("venv"), chatterbox_script, "temp_chatterbox.py")
    
    if success:
        print(f"✅ Speech generated: {output_path}")
        return output_path
    else:
        print(f"❌ Speech generation failed: {stderr}")
        return None

def main():
    """Demo combining both Flux and Chatterbox."""
    
    print("🎨 Flux + Chatterbox Integration Demo")
    print("=" * 40)
    
    # Generate image
    prompt = "A cozy library with warm lighting, digital art"
    image_path = generate_image_with_flux(prompt)
    
    # Generate speech description
    if image_path:
        description = f"I have generated an image showing {prompt}"
        speech_path = generate_speech_with_chatterbox(description)
        
        print("\n🎉 Demo completed!")
        print(f"📸 Image: {image_path}")
        print(f"🔊 Speech: {speech_path}")
    else:
        print("❌ Demo failed - could not generate image")

if __name__ == "__main__":
    main()
```

## Usage Instructions

### For Flux Only:
```powershell
# Activate Flux environment
.\activate_flux.bat

# Generate image with mflux (if installed)
mflux-generate --model schnell --prompt "A beautiful landscape" --steps 4 -q 4 --output landscape.png

# Or use Python script
python test_flux.py
```

### For Chatterbox Only:
```powershell
# Activate Chatterbox environment
.\activate_chatterbox.bat

# Use your existing Chatterbox scripts
python example_tts.py
```

### For Combined Usage:
```powershell
# Run from base directory (no venv activation needed)
python flux_chatterbox_demo.py
```

## Memory Management Tips

### For Systems with Limited VRAM:

1. **Use Quantized Models**:
   ```python
   # Use GGUF quantized models for Flux
   # Q4_1 version uses ~6.7GB instead of 22GB
   ```

2. **Sequential Usage**:
   ```python
   # Clear GPU memory between operations
   import torch
   torch.cuda.empty_cache()
   ```

3. **CPU Fallback**:
   ```python
   # Force CPU usage if needed
   device = "cpu"  # instead of "cuda"
   ```

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Use quantized models (Q4_1, Q8)
   - Reduce image resolution
   - Use CPU for one of the models

2. **Environment Conflicts**:
   - Always activate the correct environment
   - Check PyTorch versions: `python -c "import torch; print(torch.__version__)"`

3. **Model Download Issues**:
   - Ensure stable internet connection
   - Models are large (6-22GB)
   - Use Hugging Face token for gated models

### Verification Commands:

```powershell
# Check Chatterbox environment
.\activate_chatterbox.bat
python -c "import torch; print('Chatterbox PyTorch:', torch.__version__)"

# Check Flux environment  
.\activate_flux.bat
python -c "import torch; print('Flux PyTorch:', torch.__version__)"
```

## Future Notes for Yourself

### Key Points to Remember:

1. **Never mix environments**: Always activate the correct venv before running scripts
2. **PyTorch compatibility**: Both use 2.6.0, so shared CUDA drivers work
3. **Memory management**: Flux is memory-hungry, use quantized models
4. **Model storage**: Flux models are stored in `~/.cache/huggingface/`
5. **Updates**: Update each environment separately to avoid conflicts

### Recommended Workflow:

1. **Development**: Use separate terminals for each environment
2. **Production**: Use the combined script for automated workflows
3. **Debugging**: Test each system independently first
4. **Backup**: Keep working configurations documented

### Performance Optimization:

1. **Flux Schnell**: Faster, 4 steps, Apache license
2. **Flux Dev**: Higher quality, 20+ steps, non-commercial license
3. **GGUF Quantization**: Use Q4_1 for best size/quality balance
4. **Batch Processing**: Generate multiple images/audio files in sequence

## Model Recommendations

### For Limited VRAM (8-16GB):
- Flux Schnell GGUF Q4_1 (~6.7GB)
- Resolution: 512x512 or 768x768

### For High VRAM (24GB+):
- Flux Dev full model (~22GB)
- Resolution: up to 1024x1024

### For CPU Only:
- Flux Schnell with CPU fallback
- Expect much slower generation times

---

**Remember**: This setup preserves your working Chatterbox installation while adding Flux capabilities. Both systems can be used independently or together without conflicts. 