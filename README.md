# Gunshot Detection System

An advanced machine learning-based system for detecting and classifying gunshot sounds in audio streams using neural networks and acoustic analysis.

## Overview

This project implements a real-time gunshot detection system using audio processing techniques and machine learning models to identify gunshots, firearms, and related sounds. It provides both a graphical user interface and web interface for monitoring and analysis.

## Features

- **Real-time Audio Processing**: Continuous monitoring of audio input through microphone
- **Multi-level Sound Detection**:
  - Primary detection of gunshots, machine guns, and cap guns using Custom CNN model
  - Secondary detection of related sounds (explosions, impacts, mechanical sounds)
- **Advanced GUI Interface**:
  - Real-time visualization of detected sounds
  - Confidence level indicators
  - Historical detection logging
- **Web Interface**:
  - Remote monitoring capabilities
  - Detection history and statistics
  - API endpoints for integration with other systems
- **Comprehensive Logging System**:
  - Detailed event logging with timestamps
  - Sound classification confidence levels
  - Session-based log files
- **YAMNet Integration**:
  - Utilizes Google's YAMNet model for initial sound classification
  - Custom thresholds for different sound categories
  - Real-time audio processing and analysis

## System Requirements

- Python 3.8 or higher
- Windows/Linux/MacOS
- Microphone input device
- Minimum 4GB RAM
- Internet connection (for initial model download)

## Installation

1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Follow setup instructions in INSTALLATION_GUIDE.md

## Project Structure

```
gunshotdetection/
├── yamnet_detector.py          # Core YAMNet detection implementation
├── yamnet_detector_gui.py      # GUI implementation with YAMNet
├── combined_detector_gui.py    # Enhanced GUI with additional features
├── gunshot_detector_gui.py     # Main GUI application
├── web_server.py               # Web interface server
├── requirements.txt            # Project dependencies
├── src/                        # Source code modules
├── models/                     # Model files
├── yamnet_logs/                # Directory for YAMNet detection logs
└── detection_logs/             # Directory for general detection logs
```

## Usage

### GUI Application

Run the main GUI application:
```
python gunshot_detector_gui.py
```

### Combined Detector GUI

Run the enhanced GUI with combined detection methods:
```
python combined_detector_gui.py
```

### Web Interface

Run the web server for remote monitoring:
```
python web_server.py
```

## Models

The system uses several models for sound detection and classification:

1. **YAMNet** - Google's pre-trained audio event classification model
2. **Custom Classifier** - Fine-tuned model specifically for gunshot detection
3. **Neural Network** - Additional model for improving detection accuracy

## Configuration

The system includes several configurable parameters:

### Sound Detection Parameters:
```python
GUNSHOT_KEYWORDS = {
    'Machine gun': 10.0,
    'Gunshot, gunfire': 10.0,
    'Explosion': 10.0,
    'Cap gun': 10.0,
    'Burst, pop': 10.0,
    'Artillery fire': 10.0,
    'Firearms': 10.0,
    'Shot, gunshot': 10.0,
    'Ricochet': 10.0
}
```

## Logging

The system maintains detailed logs of all detections, including:
- Timestamp of detection
- Sound type and classification
- Confidence level
- Audio characteristics

## License

MIT 
