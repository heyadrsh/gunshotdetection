import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import queue
import time
import librosa
import csv
import io
import torch
import torchaudio
from urllib.request import urlopen
from threading import Thread
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QPalette

# Define gunshot-related keywords and their minimum confidence thresholds
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

class AudioProcessor(QThread):
    gunshot_detected = pyqtSignal(list)  # Signal for YAMNet detections
    gun_type_detected = pyqtSignal(str)  # Signal for gun type classification
    current_sounds = pyqtSignal(list)    # Signal for current sound detections

    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.running = False
        self.window_samples = int(self.sample_rate * 1.0)  # 1 second window
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.last_gunshot_time = 0
        self.alert_cooldown = 1.0  # 1 second cooldown between alerts

        # Load YAMNet model
        print("Loading YAMNet model...")
        tf.get_logger().setLevel('ERROR')
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Load class names
        class_map_path = urlopen("https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv")
        self.class_names = []
        csv_text = io.TextIOWrapper(class_map_path, encoding='utf-8')
        reader = csv.DictReader(csv_text)
        for row in reader:
            self.class_names.append(row['display_name'])
        
        # Load your trained model
        try:
            self.gun_classifier = torch.load('gunshot_classifier.pth')
            self.gun_classifier.eval()
        except Exception as e:
            print(f"Warning: Could not load gun classifier model: {e}")
            self.gun_classifier = None

        print("Models loaded successfully")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Error in audio recording: {status}")
        if self.running:
            self.audio_queue.put(indata.copy())

    def process_audio(self, audio):
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return librosa.util.normalize(audio)

    def check_for_gunshot(self, class_names, scores):
        detected_keywords = []
        for keyword, threshold in GUNSHOT_KEYWORDS.items():
            if keyword in class_names:
                idx = class_names.index(keyword)
                confidence = scores[idx] * 100
                if confidence >= threshold:
                    detected_keywords.append(f"{keyword}: {confidence:.1f}%")
        return detected_keywords

    def classify_gun_type(self, audio):
        if self.gun_classifier is None:
            return "Gun classifier model not loaded"
        
        try:
            # Preprocess audio for your model
            # Note: Adjust this preprocessing according to your model's requirements
            audio_tensor = torch.from_numpy(audio).float()
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                output = self.gun_classifier(audio_tensor)
                # Adjust this part based on your model's output format
                predicted_class = torch.argmax(output).item()
                
                # Map the class index to gun type name
                gun_types = ['Handgun', 'Rifle', 'Shotgun', 'Submachine Gun']  # Adjust based on your classes
                return gun_types[predicted_class]
        except Exception as e:
            return f"Error in gun classification: {e}"

    def run(self):
        self.running = True
        
        with sd.InputStream(channels=1,
                          samplerate=self.sample_rate,
                          callback=self.audio_callback,
                          blocksize=int(self.sample_rate * 0.1)):
            while self.running:
                try:
                    if not self.audio_queue.empty():
                        audio_data = self.audio_queue.get()
                        
                        # Update buffer with new audio
                        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
                        self.audio_buffer[-len(audio_data):] = audio_data.flatten()
                        
                        # Process audio
                        processed_audio = self.process_audio(self.audio_buffer)
                        
                        # YAMNet detection
                        scores, embeddings, spectrogram = self.yamnet_model(processed_audio)
                        class_scores = scores.numpy().mean(axis=0)
                        
                        # Check for gunshot sounds
                        gunshot_detections = self.check_for_gunshot(self.class_names, class_scores)
                        current_time = time.time()
                        
                        if gunshot_detections and (current_time - self.last_gunshot_time) >= self.alert_cooldown:
                            self.gunshot_detected.emit(gunshot_detections)
                            self.last_gunshot_time = current_time
                            
                            # If gunshot detected, classify the gun type
                            gun_type = self.classify_gun_type(processed_audio)
                            self.gun_type_detected.emit(gun_type)
                        
                        # Get top 5 current sounds
                        top_classes = np.argsort(class_scores)[-5:][::-1]
                        results = []
                        for idx in top_classes:
                            confidence = class_scores[idx] * 100
                            if confidence > 5.0:
                                results.append(f"{self.class_names[idx]}: {confidence:.1f}%")
                        self.current_sounds.emit(results)
                        
                except Exception as e:
                    print(f"Error in audio processing: {e}")
                time.sleep(0.01)

    def stop(self):
        self.running = False

class DetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gunshot Detection System")
        self.setGeometry(100, 100, 800, 600)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create status labels
        self.status_label = QLabel("Monitoring for gunshots...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont('Arial', 14))
        layout.addWidget(self.status_label)
        
        # Create detection display
        self.detection_text = QTextEdit()
        self.detection_text.setReadOnly(True)
        self.detection_text.setFont(QFont('Courier', 12))
        layout.addWidget(self.detection_text)
        
        # Create gun type label
        self.gun_type_label = QLabel("Gun Type: Not detected")
        self.gun_type_label.setAlignment(Qt.AlignCenter)
        self.gun_type_label.setFont(QFont('Arial', 12))
        layout.addWidget(self.gun_type_label)
        
        # Create current sounds display
        self.current_sounds_label = QLabel("Current Sounds:")
        layout.addWidget(self.current_sounds_label)
        self.current_sounds_text = QTextEdit()
        self.current_sounds_text.setReadOnly(True)
        self.current_sounds_text.setMaximumHeight(100)
        layout.addWidget(self.current_sounds_text)
        
        # Create control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)
        
        # Connect buttons
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        self.audio_processor.gunshot_detected.connect(self.on_gunshot_detected)
        self.audio_processor.gun_type_detected.connect(self.on_gun_type_detected)
        self.audio_processor.current_sounds.connect(self.update_current_sounds)
        
        # Set up log directory
        self.logs_dir = "detection_logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        self.log_filename = os.path.join(self.logs_dir, 
                                       f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def start_detection(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Monitoring for gunshots...")
        self.detection_text.clear()
        self.audio_processor.start()
        
        # Initialize log file
        with open(self.log_filename, 'w') as f:
            f.write("=== Gunshot Detection Log ===\n")
            f.write(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def stop_detection(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Detection stopped")
        self.audio_processor.stop()
        
        # Log session end
        with open(self.log_filename, 'a') as f:
            f.write(f"\nSession ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def on_gunshot_detected(self, detections):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert_text = f"\n[{timestamp}] !!! GUNSHOT DETECTED !!!\n"
        alert_text += "Detected sounds that triggered the alert:\n"
        for detection in detections:
            alert_text += f"- {detection}\n"
        alert_text += "-" * 50 + "\n"
        
        self.detection_text.append(alert_text)
        self.detection_text.verticalScrollBar().setValue(
            self.detection_text.verticalScrollBar().maximum()
        )
        
        # Log detection
        with open(self.log_filename, 'a') as f:
            f.write(alert_text)

    def on_gun_type_detected(self, gun_type):
        self.gun_type_label.setText(f"Gun Type: {gun_type}")
        
        # Log gun type
        with open(self.log_filename, 'a') as f:
            f.write(f"Classified Gun Type: {gun_type}\n")

    def update_current_sounds(self, sounds):
        self.current_sounds_text.setText(" | ".join(sounds))

    def closeEvent(self, event):
        self.audio_processor.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = DetectorGUI()
    window.show()
    sys.exit(app.exec_()) 