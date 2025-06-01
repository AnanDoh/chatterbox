import torch
from diffusers import FluxPipeline
import os

def test_flux_installation():
    """Test Flux installation and generate a sample image."""
    
    print("🎨 Testing Flux installation...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        print("\n📥 Loading Flux Schnell model...")
        print("Note: First run will download ~6GB model - this may take a while!")
        
        # Use Flux Schnell (faster, Apache 2.0 license)
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            print("✅ Model loaded on GPU")
        else:
            print("⚠️ Model loaded on CPU (will be slower)")
        
        # Generate test image
        print("\n🖼️ Generating test image...")
        prompt = "A beautiful sunset over mountains, digital art style, vibrant colors"
        
        image = pipe(
            prompt,
            num_inference_steps=4,  # Schnell works well with 4 steps
            guidance_scale=0.0,     # Schnell doesn't need guidance
            height=512,
            width=512
        ).images[0]
        
        # Save image
        output_path = "flux_test_output.png"
        image.save(output_path)
        
        print(f"✅ Flux test successful!")
        print(f"📸 Image saved as: {output_path}")
        print(f"📝 Prompt used: {prompt}")
        
        # Show file size
        if os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"📊 File size: {size_kb:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Flux test failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure you have internet connection for model download")
        print("2. Check if you have enough disk space (~6GB)")
        print("3. Try running with CPU only if GPU memory is insufficient")
        print("4. Make sure you're in the correct virtual environment")
        return False

def test_memory_usage():
    """Test different memory configurations."""
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available - skipping memory tests")
        return
    
    print("\n🧠 Memory usage information:")
    
    # Check available memory
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated() / 1024**3
    cached_memory = torch.cuda.memory_reserved() / 1024**3
    
    print(f"Total GPU memory: {total_memory:.1f} GB")
    print(f"Allocated memory: {allocated_memory:.1f} GB")
    print(f"Cached memory: {cached_memory:.1f} GB")
    print(f"Available memory: {total_memory - cached_memory:.1f} GB")
    
    # Recommendations based on available memory
    available = total_memory - cached_memory
    
    if available >= 20:
        print("✅ Sufficient memory for Flux Dev (full model)")
    elif available >= 8:
        print("✅ Sufficient memory for Flux Schnell")
        print("💡 Consider using quantized models for better performance")
    elif available >= 4:
        print("⚠️ Limited memory - use quantized models (Q4_1)")
        print("💡 Consider reducing image resolution")
    else:
        print("❌ Very limited memory - consider CPU mode")

if __name__ == "__main__":
    print("🚀 Flux Installation Test")
    print("=" * 50)
    
    # Test memory first
    test_memory_usage()
    
    # Test Flux installation
    success = test_flux_installation()
    
    if success:
        print("\n🎉 All tests passed! Flux is ready to use.")
        print("\n📚 Next steps:")
        print("1. Try different prompts with test_flux.py")
        print("2. Install mflux for easier command-line usage")
        print("3. Explore quantized models for better performance")
    else:
        print("\n❌ Tests failed. Please check the troubleshooting tips above.") 