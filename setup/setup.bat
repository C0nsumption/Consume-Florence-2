@echo off
rem Exit immediately if a command exits with a non-zero status
setlocal enabledelayedexpansion

echo Creating and navigating to project directory...
git clone https://github.com/C0nsumption/Consume-Florence-2.git
cd Consume-Florence-2

echo Setting up a virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing Git LFS...
git lfs install

echo Cloning the model repository...
echo TAKES A WHILE IF SLOW INTERNET...
git clone https://huggingface.co/microsoft/Florence-2-large

echo Installing dependencies...
pip install wheel setuptools pip --upgrade
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash_attn

echo Running tests...
python test\test_cli.py

echo Setup complete!

pause