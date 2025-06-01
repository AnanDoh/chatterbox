@echo off
echo 🎨 Flux Installation Script for Chatterbox Integration
echo =====================================================
echo.
echo This script will install Flux for image generation alongside your existing Chatterbox setup.
echo Both systems will work independently without conflicts.
echo.

pause

echo 📁 Creating Flux virtual environment...
python -m venv flux_venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment created successfully
echo.

echo 🔧 Activating Flux environment and installing dependencies...
call flux_venv\Scripts\activate.bat

echo 📦 Upgrading pip...
python -m pip install --upgrade pip

echo 📦 Installing PyTorch 2.6.0 with CUDA support...
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ⚠️ CUDA installation failed, trying CPU version...
    pip install torch==2.6.0 torchaudio==2.6.0
)

echo 📦 Installing Flux dependencies...
pip install diffusers==0.29.0
pip install transformers==4.46.3
pip install accelerate
pip install safetensors==0.5.3
pip install Pillow
pip install "numpy~=1.26.0"

echo 📦 Installing optional dependencies...
pip install gguf
pip install mflux

echo ✅ Installation completed!
echo.

echo 🧪 Testing Flux installation...
python test_flux.py

echo.
echo 🎉 Flux installation complete!
echo.
echo 📚 Usage instructions:
echo   1. Use activate_flux.bat to activate Flux environment
echo   2. Use activate_chatterbox.bat to activate Chatterbox environment
echo   3. Run python test_flux.py to test Flux
echo   4. Check FLUX_CHATTERBOX_INTEGRATION_GUIDE.md for detailed usage
echo.

pause 