"""
Flux Local Setup - Run Flux without Hugging Face dependencies
This script helps you set up and run Flux models completely locally.
"""

import os
import torch
from pathlib import Path
import requests
from tqdm import tqdm

def download_file(url, local_path, chunk_size=8192):
    """Download a file with progress bar."""
    
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f, tqdm(
        desc=local_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✅ Downloaded: {local_path}")

def setup_local_models_directory():
    """Create local models directory structure."""
    
    models_dir = Path("local_models")
    models_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (models_dir / "flux-schnell").mkdir(exist_ok=True)
    (models_dir / "flux-dev").mkdir(exist_ok=True)
    (models_dir / "gguf").mkdir(exist_ok=True)
    
    print(f"📁 Created models directory: {models_dir.absolute()}")
    return models_dir

def download_gguf_models():
    """Download GGUF quantized models (smaller, faster)."""
    
    models_dir = setup_local_models_directory()
    gguf_dir = models_dir / "gguf"
    
    # GGUF models from various sources
    gguf_models = {
        "flux1-dev-Q4_1.gguf": {
            "url": "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_1.gguf",
            "size": "6.7GB",
            "description": "Flux Dev Q4_1 quantized (good quality, smaller size)"
        },
        "flux1-schnell-Q4_1.gguf": {
            "url": "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_1.gguf", 
            "size": "6.7GB",
            "description": "Flux Schnell Q4_1 quantized (faster inference)"
        }
    }
    
    print("🔽 Available GGUF models for download:")
    for name, info in gguf_models.items():
        print(f"  📦 {name} ({info['size']}) - {info['description']}")
    
    choice = input("\nWhich model would you like to download? (schnell/dev/both/skip): ").lower()
    
    if choice in ["schnell", "both"]:
        model_path = gguf_dir / "flux1-schnell-Q4_1.gguf"
        if not model_path.exists():
            print(f"\n📥 Downloading Flux Schnell GGUF...")
            try:
                download_file(gguf_models["flux1-schnell-Q4_1.gguf"]["url"], model_path)
            except Exception as e:
                print(f"❌ Download failed: {e}")
    
    if choice in ["dev", "both"]:
        model_path = gguf_dir / "flux1-dev-Q4_1.gguf"
        if not model_path.exists():
            print(f"\n📥 Downloading Flux Dev GGUF...")
            try:
                download_file(gguf_models["flux1-dev-Q4_1.gguf"]["url"], model_path)
            except Exception as e:
                print(f"❌ Download failed: {e}")

def create_local_flux_test():
    """Create a test script that uses local models."""
    
    test_script = '''
import torch
from diffusers import FluxPipeline
from pathlib import Path
import os

def test_local_flux():
    """Test Flux with local models."""
    
    print("🎨 Testing Local Flux Setup...")
    
    # Check for local models
    models_dir = Path("local_models")
    
    if not models_dir.exists():
        print("❌ No local_models directory found!")
        print("💡 Run flux_local_setup.py first to download models")
        return False
    
    # Look for GGUF models
    gguf_dir = models_dir / "gguf"
    gguf_models = list(gguf_dir.glob("*.gguf")) if gguf_dir.exists() else []
    
    if gguf_models:
        print(f"✅ Found {len(gguf_models)} GGUF model(s)")
        for model in gguf_models:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"  📦 {model.name} ({size_mb:.1f} MB)")
    
    # Try to load a model locally (this is a simplified example)
    # Note: GGUF models require special loaders
    print("\\n💡 For GGUF models, you'll need ComfyUI or specialized loaders")
    print("💡 For standard models, download them to local_models/flux-schnell/ or local_models/flux-dev/")
    
    return True

if __name__ == "__main__":
    test_local_flux()
'''
    
    with open("test_local_flux.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created test_local_flux.py")

def create_offline_flux_script():
    """Create a script that works completely offline."""
    
    offline_script = '''
"""
Offline Flux Image Generator
Works without internet connection once models are downloaded
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

class OfflineFluxSimulator:
    """
    A simple offline image generator that simulates Flux-style outputs
    without requiring actual Flux models (useful for testing workflows)
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎨 Offline Flux Simulator initialized on {self.device}")
    
    def generate_image(self, prompt, width=512, height=512, steps=4):
        """Generate a procedural image based on the prompt."""
        
        print(f"🖼️ Generating image for: '{prompt}'")
        print(f"📐 Size: {width}x{height}, Steps: {steps}")
        
        # Create base image
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        # Generate colors based on prompt keywords
        colors = self._get_colors_from_prompt(prompt)
        
        # Create gradient background
        for y in range(height):
            color_ratio = y / height
            color = self._blend_colors(colors[0], colors[1], color_ratio)
            draw.line([(0, y), (width, y)], fill=color)
        
        # Add shapes based on prompt
        self._add_shapes_from_prompt(draw, prompt, width, height)
        
        # Add text overlay
        try:
            font = ImageFont.load_default()
            text_y = height - 30
            draw.text((10, text_y), f"Generated: {prompt[:30]}...", fill='white', font=font)
        except:
            draw.text((10, height - 20), "Generated Image", fill='white')
        
        return img
    
    def _get_colors_from_prompt(self, prompt):
        """Extract color scheme from prompt."""
        
        color_map = {
            'sunset': [(255, 165, 0), (255, 69, 0)],
            'ocean': [(0, 119, 190), (0, 180, 216)],
            'forest': [(34, 139, 34), (0, 100, 0)],
            'mountain': [(139, 137, 137), (169, 169, 169)],
            'sky': [(135, 206, 235), (176, 196, 222)],
            'fire': [(255, 69, 0), (255, 140, 0)],
            'night': [(25, 25, 112), (72, 61, 139)],
        }
        
        prompt_lower = prompt.lower()
        for keyword, colors in color_map.items():
            if keyword in prompt_lower:
                return colors
        
        # Default colors
        return [(100, 150, 200), (200, 150, 100)]
    
    def _blend_colors(self, color1, color2, ratio):
        """Blend two colors."""
        return tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2))
    
    def _add_shapes_from_prompt(self, draw, prompt, width, height):
        """Add shapes based on prompt content."""
        
        prompt_lower = prompt.lower()
        
        if 'circle' in prompt_lower or 'sun' in prompt_lower:
            # Add circles
            for _ in range(random.randint(1, 3)):
                x = random.randint(0, width)
                y = random.randint(0, height)
                r = random.randint(20, 80)
                color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline='white', width=2)
        
        if 'mountain' in prompt_lower:
            # Add triangular shapes
            for _ in range(random.randint(2, 5)):
                x1 = random.randint(0, width)
                y1 = height
                x2 = x1 + random.randint(-100, 100)
                y2 = random.randint(height//3, height//2)
                x3 = x1 + random.randint(-100, 100)
                y3 = height
                color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
                draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=color, outline='white')

def main():
    """Demo the offline flux simulator."""
    
    simulator = OfflineFluxSimulator()
    
    prompts = [
        "A beautiful sunset over mountains",
        "Ocean waves under a starry night sky",
        "Forest with morning sunlight",
        "Abstract circles in vibrant colors"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\\n🎨 Generating image {i+1}/{len(prompts)}")
        
        image = simulator.generate_image(prompt)
        output_path = f"offline_flux_output_{i+1}.png"
        image.save(output_path)
        
        print(f"✅ Saved: {output_path}")
    
    print("\\n🎉 Offline generation complete!")
    print("💡 This demonstrates the workflow without requiring actual Flux models")

if __name__ == "__main__":
    main()
'''
    
    with open("offline_flux_demo.py", "w") as f:
        f.write(offline_script)
    
    print("✅ Created offline_flux_demo.py")

def main():
    """Main setup function."""
    
    print("🚀 Flux Local Setup")
    print("=" * 50)
    print("This script helps you run Flux completely locally without Hugging Face")
    print()
    
    options = {
        "1": ("Download GGUF Models", download_gguf_models),
        "2": ("Create Local Test Script", create_local_flux_test),
        "3": ("Create Offline Demo", create_offline_flux_script),
        "4": ("Setup All", lambda: [download_gguf_models(), create_local_flux_test(), create_offline_flux_script()]),
    }
    
    print("📋 Available options:")
    for key, (desc, _) in options.items():
        print(f"  {key}. {desc}")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice in options:
        print(f"\n🔧 Running: {options[choice][0]}")
        try:
            options[choice][1]()
            print("\n✅ Setup complete!")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print("❌ Invalid choice")
    
    print("\n📚 Next steps:")
    print("1. Use offline_flux_demo.py for immediate testing")
    print("2. Download GGUF models for actual Flux inference")
    print("3. Set up ComfyUI for advanced GGUF model usage")

if __name__ == "__main__":
    main() 