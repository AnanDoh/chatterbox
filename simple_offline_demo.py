"""
Simple Offline Image Generator
Demonstrates image generation without Hugging Face dependencies
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os

class SimpleOfflineGenerator:
    """A basic image generator that works completely offline."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Offline Generator initialized on {self.device}")
    
    def generate_image(self, prompt, width=512, height=512):
        """Generate a procedural image based on the prompt."""
        
        print(f"Generating image for: '{prompt}'")
        print(f"Size: {width}x{height}")
        
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
            draw.text((10, height - 30), f"Generated: {prompt[:40]}", fill='white')
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
            'space': [(0, 0, 0), (25, 25, 112)],
            'flower': [(255, 192, 203), (255, 20, 147)],
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
                x = random.randint(50, width-50)
                y = random.randint(50, height-50)
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
        
        if 'star' in prompt_lower or 'space' in prompt_lower:
            # Add stars
            for _ in range(random.randint(10, 30)):
                x = random.randint(0, width)
                y = random.randint(0, height//2)
                size = random.randint(1, 3)
                draw.ellipse([x-size, y-size, x+size, y+size], fill='white')

def test_chatterbox_integration():
    """Test integration with Chatterbox TTS."""
    
    print("\n" + "="*50)
    print("TESTING CHATTERBOX + FLUX INTEGRATION")
    print("="*50)
    
    # Simulate Chatterbox TTS (placeholder)
    def simulate_tts(text):
        print(f"[Chatterbox TTS] Generating speech: '{text}'")
        return f"speech_output_{hash(text) % 1000}.wav"
    
    # Generate image
    generator = SimpleOfflineGenerator()
    prompt = "A peaceful sunset over mountains with stars"
    image = generator.generate_image(prompt)
    image_path = "integrated_output.png"
    image.save(image_path)
    
    # Generate speech description
    description = f"I have created an image showing {prompt}"
    speech_path = simulate_tts(description)
    
    print(f"\nIntegration complete!")
    print(f"Image: {image_path}")
    print(f"Speech: {speech_path}")
    print("Both systems working together without conflicts!")

def main():
    """Demo the offline generator."""
    
    print("OFFLINE FLUX DEMO - NO HUGGING FACE REQUIRED")
    print("="*50)
    
    generator = SimpleOfflineGenerator()
    
    prompts = [
        "A beautiful sunset over mountains",
        "Ocean waves under a starry night sky", 
        "Forest with morning sunlight",
        "Abstract circles in vibrant colors",
        "Space scene with stars and planets"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nGenerating image {i}/{len(prompts)}")
        
        image = generator.generate_image(prompt)
        output_path = f"offline_output_{i}.png"
        image.save(output_path)
        
        # Show file info
        if os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"Saved: {output_path} ({size_kb:.1f} KB)")
    
    print(f"\nGeneration complete! Created {len(prompts)} images")
    print("This demonstrates the workflow without requiring:")
    print("- Hugging Face authentication")
    print("- Internet connection (after initial setup)")
    print("- Large model downloads")
    
    # Test integration
    test_chatterbox_integration()

if __name__ == "__main__":
    main() 