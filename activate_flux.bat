@echo off
echo Activating Flux environment...
call flux_venv\Scripts\activate.bat
echo Flux environment activated. PyTorch version:
python -c "import torch; print(torch.__version__)"
echo.
echo You can now use Flux commands:
echo   python test_flux.py
echo   mflux-generate --model schnell --prompt "your prompt" --steps 4 -q 4 --output image.png
echo. 