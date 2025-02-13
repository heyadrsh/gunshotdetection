@echo off
setlocal EnableDelayedExpansion

:: Set environment variables to suppress TensorFlow warnings
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
set PYTHONWARNINGS=ignore::UserWarning
set CUDA_VISIBLE_DEVICES=-1

echo Starting Comprehensive Gunshot Detection System Setup...
echo =====================================

:: Function to check Python version
:check_python_version
for /f "tokens=2 delims=." %%I in ('python -V 2^>^&1 ^| find "Python"') do set PYTHON_VERSION=%%I
if "%PYTHON_VERSION%"=="10" (
    echo Found compatible Python 3.10
    goto :python_found
)

:: Check if Python 3.10 is installed elsewhere
where python3.10 >nul 2>&1
if %errorLevel% equ 0 (
    echo Found Python 3.10, setting as active version
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python310;%LOCALAPPDATA%\Programs\Python\Python310\Scripts;%PATH%"
    goto :python_found
)

:: Install Python 3.10 if not found
echo Python 3.10 not found. Installing for current user...
powershell -Command "& { Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe' -OutFile 'python_installer.exe' }"
python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 DefaultAllUsersTargetDir="%LOCALAPPDATA%\Programs\Python\Python310"
del python_installer.exe
echo Python 3.10 installed successfully.

:python_found
:: Verify Python installation
python -V
if errorlevel 1 (
    echo Failed to verify Python installation
    pause
    exit /b 1
)

:: Check for C/C++ build tools
echo Checking for C/C++ build tools...
gcc --version >nul 2>&1
if %errorLevel% neq 0 (
    echo C/C++ build tools (MinGW) not found.
    choice /C YN /M "Do you want to install MinGW"
    if errorlevel 2 (
        echo Skipping MinGW installation. Assuming build tools are installed...
    ) else (
        echo Installing MinGW for current user...
        :: Create MinGW directory in local app data
        set "MINGW_PATH=%LOCALAPPDATA%\MinGW64"
        if not exist "!MINGW_PATH!" mkdir "!MINGW_PATH!"
        
        :: Download and extract MinGW
        echo Downloading MinGW...
        powershell -Command "& { Invoke-WebRequest -Uri 'https://github.com/brechtsanders/winlibs_mingw/releases/download/13.2.0-16.0.6-11.0.1-ucrt-r1/winlibs-x86_64-posix-seh-gcc-13.2.0-mingw-w64ucrt-11.0.1-r1.zip' -OutFile 'mingw.zip' }"
        
        echo Extracting MinGW...
        powershell -Command "Expand-Archive -Path mingw.zip -DestinationPath '!MINGW_PATH!' -Force"
        if exist mingw.zip del mingw.zip
        
        :: Update PATH without using setx
        set "USER_PATH="
        for /f "skip=2 tokens=1,2*" %%N in ('reg query HKCU\Environment /v PATH 2^>nul') do if /i "%%N" == "PATH" set "USER_PATH=%%P"
        
        if "!USER_PATH!" == "" (
            reg add HKCU\Environment /v PATH /t REG_EXPAND_SZ /d "!MINGW_PATH!\mingw64\bin" /f
        ) else (
            reg add HKCU\Environment /v PATH /t REG_EXPAND_SZ /d "!USER_PATH!;!MINGW_PATH!\mingw64\bin" /f
        )
        
        :: Update current session PATH
        set "PATH=!PATH!;!MINGW_PATH!\mingw64\bin"
        
        echo MinGW installed successfully.
        echo NOTE: You may need to restart your terminal for the PATH changes to take effect.
    )
)

:: Create and activate virtual environment
echo Creating virtual environment...
if exist venv310 (
    echo Virtual environment already exists.
) else (
    python -m venv venv310
)

:: Activate virtual environment and set environment variables in venv
call venv310\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment!
    pause
    exit /b 1
)

:: Set environment variables in virtual environment activation script
echo set TF_ENABLE_ONEDNN_OPTS=0 >> venv310\Scripts\activate.bat
echo set TF_CPP_MIN_LOG_LEVEL=2 >> venv310\Scripts\activate.bat
echo set PYTHONWARNINGS=ignore::UserWarning >> venv310\Scripts\activate.bat
echo set CUDA_VISIBLE_DEVICES=-1 >> venv310\Scripts\activate.bat

:: Create necessary directories
echo Creating necessary directories...
if not exist yamnet_logs mkdir yamnet_logs
if not exist detection_logs mkdir detection_logs
if not exist sound_detection_logs mkdir sound_detection_logs
if not exist models mkdir models

:: Upgrade pip first
python -m pip install --upgrade pip

:: Install all required packages with specific versions
echo Installing dependencies...

:: Core ML packages
pip install numpy==1.24.3
pip install tensorflow==2.13.0
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

:: Audio processing packages
pip install sounddevice==0.4.6
pip install librosa==0.10.1
pip install resampy==0.4.3
pip install colorednoise==2.2.0

:: GUI packages
pip install PyQt6==6.8.0

:: Data processing and visualization
pip install matplotlib==3.7.1
pip install pandas==2.0.3
pip install seaborn==0.13.2
pip install tensorflow-hub==0.13.0
pip install tqdm==4.67.1

:: Download required models
echo Downloading required models...
if not exist "models/yamnet.h5" (
    echo Downloading YAMNet model...
    powershell -Command "& { Invoke-WebRequest -Uri 'https://storage.googleapis.com/audioset/yamnet.h5' -OutFile 'models/yamnet.h5' }"
)

:: Verify installations
echo Verifying installations...
python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__)"
python -c "import torch; print('PyTorch Version:', torch.__version__)"
python -c "import sounddevice as sd; print('Sounddevice Version:', sd.__version__)"

:: Check for model file
echo Checking for model files...
if not exist gunshot_classifier.pth (
    echo Warning: gunshot_classifier.pth not found!
    echo Please ensure the model file is present in the workspace.
)

:: Configure audio devices
echo Configuring audio devices...
python -c "import sounddevice as sd; print('Available Audio Devices:\n'); print(sd.query_devices())"

:: Create a test script to verify everything
echo Creating test script...
echo import tensorflow as tf > test_setup.py
echo import torch >> test_setup.py
echo import sounddevice as sd >> test_setup.py
echo import numpy as np >> test_setup.py
echo import librosa >> test_setup.py
echo import PyQt6 >> test_setup.py
echo import seaborn >> test_setup.py
echo import resampy >> test_setup.py
echo import tqdm >> test_setup.py
echo print('All imports successful!') >> test_setup.py

echo Testing installation...
python test_setup.py
del test_setup.py

:: Setup complete
echo.
echo =====================================
echo Setup completed successfully!
echo.
echo Available commands:
echo - python yamnet_detector_gui.py     (YAMNet GUI)
echo - python combined_gunshot_gui.py    (Combined GUI)
echo - python test_model.py              (Test Model)
echo.
echo Note: Make sure your microphone is properly connected and set as default input device.
echo If you encounter any issues, please check the following:
echo 1. Verify your audio device is properly connected
echo 2. Check Windows Sound settings
echo 3. Ensure all models are downloaded correctly
echo 4. Verify Python PATH is set correctly
echo =====================================

pause