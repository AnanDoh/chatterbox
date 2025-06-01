import torch
import sys
import os

def test_basic_imports():
    """Test if all required packages can be imported."""
    
    print("🧪 Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        from diffusers import FluxPipeline
        print("✅ FluxPipeline imported successfully")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__} imported successfully")
        
        import accelerate
        print("✅ Accelerate imported successfully")
        
        import safetensors
        print("✅ SafeTensors imported successfully")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
        
        from PIL import Image
        print("✅ Pillow imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_torch_setup():
    """Test PyTorch configuration."""
    
    print("\n🔧 Testing PyTorch setup...")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        device = "cuda"
    else:
        print("Running on CPU (slower but will work)")
        device = "cpu"
    
    # Test basic tensor operations
    try:
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = torch.matmul(x, y)
        print(f"✅ Basic tensor operations work on {device}")
        return True
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False

def test_huggingface_connection():
    """Test connection to Hugging Face without downloading large models."""
    
    print("\n🌐 Testing Hugging Face connection...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Test with a small, public model
        model_info = api.model_info("hf-internal-testing/tiny-stable-diffusion-torch")
        print("✅ Hugging Face connection successful")
        return True
        
    except Exception as e:
        print(f"⚠️ Hugging Face connection issue: {e}")
        print("💡 This might affect model downloads, but local inference should work")
        return False

def create_test_image():
    """Create a simple test image to verify PIL functionality."""
    
    print("\n🖼️ Creating test image...")
    
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create a simple test image
        img = Image.new('RGB', (512, 512), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw some shapes
        draw.rectangle([100, 100, 400, 400], fill='white', outline='black', width=3)
        draw.text((200, 250), "Flux Test", fill='black')
        
        # Save test image
        output_path = "test_image_output.png"
        img.save(output_path)
        
        print(f"✅ Test image created: {output_path}")
        
        # Show file size
        if os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"📊 File size: {size_kb:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Image creation failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("🚀 Flux Environment Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("PyTorch Setup", test_torch_setup),
        ("Hugging Face Connection", test_huggingface_connection),
        ("Image Creation", create_test_image)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Flux environment is ready.")
        print("\n📚 Next steps:")
        print("1. Set up Hugging Face authentication for model downloads")
        print("2. Try generating images with Flux models")
        print("3. Explore different model variants (Schnell vs Dev)")
    elif passed >= len(results) - 1:
        print("\n✅ Environment is mostly ready!")
        print("💡 Minor issues detected but core functionality should work")
    else:
        print("\n⚠️ Several issues detected. Please check the failed tests above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 