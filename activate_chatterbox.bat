@echo off
echo Activating Chatterbox environment...
call venv\Scripts\activate.bat
echo Chatterbox environment activated. PyTorch version:
python -c "import torch; print(torch.__version__)"
echo.
echo You can now use Chatterbox TTS commands:
echo   python example_tts.py
echo   python gradio_tts_app.py
echo. 