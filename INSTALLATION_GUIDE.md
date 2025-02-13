# Detailed Installation and Setup Guide

## Prerequisites
- Windows 10 or higher
- Python 3.8 or higher (3.8.10 recommended)
- Git (optional, for cloning)
- Internet connection (for downloading dependencies)
- Microphone device

## Step 1: Environment Setup

### 1.1 Python Installation
1. Download Python 3.8.10 from [Python Official Website](https://www.python.org/downloads/release/python-3810/)
2. Run the installer
   - ✅ Check "Add Python 3.8 to PATH"
   - ✅ Check "Install pip"
3. Verify installation:
```bash
python --version
pip --version
```

### 1.2 Project Setup
1. Create a new directory for the project:
```bash
mkdir gunshotdetection
cd gunshotdetection
```

2. Extract the project zip file into this directory

### 1.3 Virtual Environment Setup
1. Create a virtual environment:
```bash
python -m venv venv310
```

2. Activate the virtual environment:
```bash
# On Windows
venv310\Scripts\activate
```

## Step 2: Dependencies Installation

### 2.1 Install Required Packages
1. Install primary dependencies:
```bash
pip install tensorflow==2.13.0
pip install torch==2.1.0 torchaudio==2.1.0
pip install sounddevice==0.4.6
pip install librosa==0.10.1
pip install PyQt5==5.15.9
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
```

2. Install additional dependencies:
```bash
pip install matplotlib
pip install pandas
pip install tensorflow-hub
```

### 2.2 Verify Installation
Run these commands to verify installations:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import torch; print(torch.__version__)"
python -c "import sounddevice as sd; print(sd.__version__)"
```

## Step 3: Audio Setup

### 3.1 Configure Audio Device
1. Open Windows Sound Settings
2. Set your microphone as the default input device
3. Test microphone in Windows settings

### 3.2 Verify Audio Input
Run this Python code to test audio setup:
```python
import sounddevice as sd
print("Available devices:")
print(sd.query_devices())
```

## Step 4: Running the Applications

### 4.1 YAMNet Detector GUI
1. Navigate to project directory
2. Run:
```bash
python yamnet_detector_gui.py
```
Expected output:
- YAMNet model loading message
- GUI window appears
- Status shows "Ready"

### 4.2 Combined Gunshot GUI
1. Run:
```bash
python combined_gunshot_gui.py
```
Features:
- Real-time detection
- Visual alerts
- Logging system

### 4.3 Testing Trained Model
1. Run the test script:
```bash
python test_model.py
```
2. When prompted, provide path to audio file:
```
Enter audio file path: dataset/AK-47/test_sample.wav
```

### 4.4 Running Feature Extraction
1. Ensure dataset is in correct structure:
```
dataset/
├── AK-12/
├── AK-47/
├── M249/
├── MG-42/
├── MP5/
└── Zastava M92/
```

2. Run extraction:
```bash
python extract_features.py
```

## Step 5: Troubleshooting

### 5.1 Common Issues and Solutions

#### Audio Device Not Found
```bash
python -c "import sounddevice as sd; sd.query_devices()"
```
- Check if your device is listed
- Set default device if needed:
```python
import sounddevice as sd
sd.default.device = [0, 0]  # Adjust indices as needed
```

#### CUDA/GPU Issues
- For CPU-only operation, no action needed
- For GPU support, install CUDA toolkit matching your PyTorch version

#### Import Errors
- Verify all packages are installed:
```bash
pip list
```
- Reinstall problematic package:
```bash
pip uninstall package_name
pip install package_name
```

### 5.2 Performance Optimization
1. Adjust audio block size in settings:
```python
CHUNK_SIZE = 1024  # Decrease if CPU usage is high
```

2. Modify detection thresholds:
```python
CONFIDENCE_THRESHOLD = 0.1  # Increase to reduce false positives
```

## Step 6: Directory Structure
Ensure your workspace has this structure:
```
gunshotdetectionsimple/
├── yamnet_detector.py
├── yamnet_detector_gui.py
├── combined_detector_gui.py
├── test_model.py
├── extract_features.py
├── requirements.txt
├── dataset/
├── yamnet_logs/
└── detection_logs/
```

## Step 7: Verification Steps

### 7.1 System Check
Run this verification script:
```bash
python system_check.py
```

### 7.2 Test Each Component
1. Basic YAMNet detection:
```bash
python yamnet_detector.py
```

2. GUI interface:
```bash
python combined_detector_gui.py
```

3. Model testing:
```bash
python test_model.py
```

## Additional Notes

### Log File Locations
- YAMNet detection logs: `yamnet_logs/`
- General detection logs: `detection_logs/`
- Training logs: `training_logs/`

### Model Files
- YAMNet model: Downloads automatically
- Trained classifier: `gunshot_classifier.pth`

### Configuration Files
- Audio settings in respective Python files
- Threshold values in detector scripts
- GUI settings in interface files

## Support and Updates
- Check console output for error messages
- Verify Python and package versions match requirements
- Monitor system resources during execution
- Keep all dependencies updated

For any issues:
1. Check error messages
2. Verify installation steps
3. Confirm audio device setup
4. Review log files
5. Ensure correct Python version 