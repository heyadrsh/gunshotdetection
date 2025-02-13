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
from threading import Thread
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QPalette
from train_model_colab import GunShotModel
import subprocess

# YAMNet detection parameters
GUNSHOT_KEYWORDS = {
    'Gunshot, gunfire': 7.0,
    'Machine gun': 7.0,
    'Artillery fire': 7.0,
    'Cap gun': 7.0,
    'Ricochet': 7.0,
    'Shot, gunshot':7.0,
    'Firearms': 7.0
}

# Audio preprocessing parameters
SAMPLE_RATE = 22050
MAX_DURATION = 10.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

class AudioProcessor(QThread):
    gunshot_detected = pyqtSignal(list)  # Signal for YAMNet detections
    gun_type_detected = pyqtSignal(tuple)  # Signal for gun type classification (type, confidence, all_probs)
    current_sounds = pyqtSignal(list)    # Signal for current sound detections

    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.running = False
        self.window_samples = int(self.sample_rate * 2.0)  # 2 second window for detection
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.last_gunshot_time = 0
        self.alert_cooldown = 1.0  # 1 second cooldown between alerts

        # Load YAMNet model from local path
        print("Loading YAMNet model...")
        tf.get_logger().setLevel('ERROR')
        model_path = os.path.join('models', 'yamnet_model')
        if not os.path.exists(model_path):
            raise FileNotFoundError("YAMNet model not found. Please run download_yamnet_resources.py first.")
        self.yamnet_model = hub.load(model_path)
        
        # Load class names from local file
        class_map_path = os.path.join('models', 'yamnet_class_map.csv')
        if not os.path.exists(class_map_path):
            raise FileNotFoundError("Class map not found. Please run download_yamnet_resources.py first.")
        
        self.class_names = []
        with open(class_map_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.class_names.append(row['display_name'])

        # Load gun classifier model
        print("Loading gun classifier model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gun_classifier = GunShotModel(num_classes=len(np.load('label_encoder_classes.npy')))
        checkpoint = torch.load('best_model.pth', map_location=self.device)
        self.gun_classifier.load_state_dict(checkpoint['model_state_dict'])
        self.gun_classifier.to(self.device)
        self.gun_classifier.eval()
        
        # Load label encoder classes
        self.label_encoder_classes = np.load('label_encoder_classes.npy')
        
        print("Models loaded successfully")

    def preprocess_audio_for_classification(self, audio):
        """Preprocess audio for gun classification model."""
        # Resample to required sample rate
        if self.sample_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=SAMPLE_RATE)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            power=2.0
        )
        
        # Convert to dB scale and normalize
        mel_spec_db = librosa.power_to_db(mel_spec + 1e-9, ref=np.max)
        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db) + 1e-9
        mel_spec_db = (mel_spec_db - mean) / std
        
        return mel_spec_db.astype(np.float32)

    def classify_gun_type(self, audio):
        """Classify the type of gun from audio."""
        try:
            # Preprocess audio
            features = self.preprocess_audio_for_classification(audio)
            features = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.inference_mode():
                outputs = self.gun_classifier(features)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                probabilities = probabilities.cpu().numpy() * 100
                
                # Get predicted class
                predicted_class = torch.argmax(outputs, dim=1).item()
                predicted_type = self.label_encoder_classes[predicted_class]
                confidence = probabilities[predicted_class]
                
                return predicted_type, confidence, probabilities
        except Exception as e:
            print(f"Error in gun classification: {e}")
            return None, 0, []

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
                            
                            # Classify gun type using the current audio buffer
                            gun_type, confidence, all_probs = self.classify_gun_type(processed_audio)
                            self.gun_type_detected.emit((gun_type, confidence, all_probs))
                        
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
        self.setWindowTitle("Advanced Gunshot Detection System")
        self.setGeometry(100, 100, 1200, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QWidget {
                background-color: #ffffff;
                color: #2c3e50;
            }
            QLabel {
                color: #2c3e50;
            }
            QTextEdit {
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 12px;
                color: #2c3e50;
            }
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 25px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QPushButton:disabled {
                background-color: #e9ecef;
                color: #6c757d;
            }
        """)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Create header section
        header_container = QWidget()
        header_layout = QVBoxLayout(header_container)
        header_layout.setSpacing(10)
        
        # Title and status
        title_label = QLabel("Advanced Gunshot Detection System")
        title_label.setFont(QFont('Arial', 24, QFont.Bold))
        title_label.setStyleSheet("""
            color: #0d6efd;
            padding: 10px;
            text-align: center;
            border-bottom: 2px solid #e9ecef;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)
        
        self.status_label = QLabel("System Ready - Click Start to Begin Monitoring")
        self.status_label.setFont(QFont('Arial', 14))
        self.status_label.setStyleSheet("""
            color: #2c3e50;
            padding: 10px;
            background-color: #e9ecef;
            border-radius: 8px;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.status_label)
        
        layout.addWidget(header_container)
        
        # Create main content area
        content_container = QWidget()
        content_layout = QHBoxLayout(content_container)
        content_layout.setSpacing(20)
        
        # Left panel - Detection History
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
        detection_label = QLabel("Detection History")
        detection_label.setFont(QFont('Arial', 14, QFont.Bold))
        detection_label.setStyleSheet("color: #0d6efd; margin-bottom: 5px;")
        left_layout.addWidget(detection_label)
        
        self.detection_text = QTextEdit()
        self.detection_text.setReadOnly(True)
        self.detection_text.setFont(QFont('Consolas', 12))
        self.detection_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        left_layout.addWidget(self.detection_text)
        
        content_layout.addWidget(left_panel, stretch=60)
        
        # Right panel - Gun Classification & Current Sounds
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        
        # Gun Type Section
        gun_type_container = QWidget()
        gun_type_layout = QVBoxLayout(gun_type_container)
        gun_type_layout.setSpacing(10)
        
        gun_type_header = QLabel("Gun Classification")
        gun_type_header.setFont(QFont('Arial', 14, QFont.Bold))
        gun_type_header.setStyleSheet("color: #0d6efd; margin-bottom: 5px;")
        gun_type_layout.addWidget(gun_type_header)
        
        self.gun_type_label = QLabel("Awaiting Detection...")
        self.gun_type_label.setFont(QFont('Arial', 12))
        self.gun_type_label.setStyleSheet("""
            padding: 15px;
            background-color: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 8px;
        """)
        self.gun_type_label.setAlignment(Qt.AlignCenter)
        gun_type_layout.addWidget(self.gun_type_label)
        
        self.gun_probabilities = QTextEdit()
        self.gun_probabilities.setReadOnly(True)
        self.gun_probabilities.setMaximumHeight(200)
        self.gun_probabilities.setFont(QFont('Consolas', 11))
        gun_type_layout.addWidget(self.gun_probabilities)
        
        right_layout.addWidget(gun_type_container)
        
        # Current Sounds Section
        sounds_container = QWidget()
        sounds_layout = QVBoxLayout(sounds_container)
        sounds_layout.setSpacing(10)
        
        sounds_header = QLabel("Live Audio Analysis")
        sounds_header.setFont(QFont('Arial', 14, QFont.Bold))
        sounds_header.setStyleSheet("color: #0d6efd; margin-bottom: 5px;")
        sounds_layout.addWidget(sounds_header)
        
        self.current_sounds_text = QTextEdit()
        self.current_sounds_text.setReadOnly(True)
        self.current_sounds_text.setMaximumHeight(100)
        self.current_sounds_text.setFont(QFont('Consolas', 11))
        sounds_layout.addWidget(self.current_sounds_text)
        
        right_layout.addWidget(sounds_container)
        
        content_layout.addWidget(right_panel, stretch=40)
        layout.addWidget(content_container)
        
        # Control Panel
        control_container = QWidget()
        control_layout = QHBoxLayout(control_container)
        control_layout.setSpacing(20)
        
        self.start_button = QPushButton("Start Detection")
        self.stop_button = QPushButton("Stop Detection")
        
        for button in [self.start_button, self.stop_button]:
            button.setFont(QFont('Arial', 12, QFont.Bold))
            button.setMinimumHeight(50)
            button.setMinimumWidth(200)
        
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
            }
            QPushButton:hover {
                background-color: #bb2d3b;
            }
        """)
        
        control_layout.addStretch()
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()
        
        layout.addWidget(control_container)
        
        # Status bar for logging info
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #f8f9fa;
                color: #6c757d;
                border-top: 1px solid #dee2e6;
                padding: 8px;
            }
        """)
        self.statusBar().showMessage("System initialized and ready")
        
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

        # Launch neural network visualization
        self.launch_neural_network_visualization()

    def start_detection(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("[ACTIVE] Monitoring for Gunshots")
        self.status_label.setStyleSheet("""
            color: #198754;
            font-weight: bold;
            padding: 10px;
            background-color: #e8f5e9;
            border-radius: 8px;
        """)
        self.detection_text.clear()
        self.audio_processor.start()
        
        # Initialize log file with UTF-8 encoding
        with open(self.log_filename, 'w', encoding='utf-8') as f:
            f.write("=== Advanced Gunshot Detection Log ===\n")
            f.write(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def stop_detection(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("[STOPPED] Detection Stopped")
        self.status_label.setStyleSheet("""
            color: #dc3545;
            font-weight: bold;
            padding: 10px;
            background-color: #f8d7da;
            border-radius: 8px;
        """)
        self.audio_processor.stop()
        
        # Log session end with UTF-8 encoding
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(f"\nSession ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def on_gunshot_detected(self, detections):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert_text = f"\n[{timestamp}] !!! GUNSHOT DETECTED !!!\n"
        alert_text += "Detected sounds that triggered the alert:\n"
        for detection in detections:
            alert_text += f"-> {detection}\n"
        alert_text += "=" * 50 + "\n"
        
        self.detection_text.append(alert_text)
        self.detection_text.verticalScrollBar().setValue(
            self.detection_text.verticalScrollBar().maximum()
        )
        
        # Change status label to alert mode
        self.status_label.setText("[ALERT] GUNSHOT DETECTED")
        self.status_label.setStyleSheet("""
            color: #dc3545;
            font-weight: bold;
            padding: 10px;
            background-color: #f8d7da;
            border-radius: 8px;
        """)
        
        # Log detection with UTF-8 encoding
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(alert_text)

    def on_gun_type_detected(self, detection_info):
        gun_type, confidence, all_probs = detection_info
        if gun_type:
            self.gun_type_label.setText(f"[DETECTED] {gun_type} ({confidence:.1f}%)")
            self.gun_type_label.setStyleSheet("""
                color: #0d6efd;
                font-weight: bold;
                padding: 15px;
                background-color: #cfe2ff;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                margin: 10px 0;
            """)
            
            # Update probabilities display
            prob_text = "Classification Results:\n\n"
            sorted_indices = np.argsort(all_probs)[::-1]
            for idx in sorted_indices:
                gun_name = self.audio_processor.label_encoder_classes[idx]
                prob = all_probs[idx]
                prob_text += f"{'>>' if idx == sorted_indices[0] else '--'} {gun_name}: {prob:.1f}%\n"
            self.gun_probabilities.setText(prob_text)
            
            # Log gun type with UTF-8 encoding
            with open(self.log_filename, 'a', encoding='utf-8') as f:
                f.write(f"Classified Gun Type: {gun_type} (Confidence: {confidence:.1f}%)\n")
                f.write("All probabilities:\n")
                f.write(prob_text + "\n")

    def update_current_sounds(self, sounds):
        formatted_sounds = []
        for sound in sounds:
            name, confidence = sound.split(': ')
            formatted_sounds.append(f"• {name}: {confidence}")
        self.current_sounds_text.setText("\n".join(formatted_sounds))

    def closeEvent(self, event):
        self.audio_processor.stop()
        self.statusBar().showMessage("Shutting down...")
        event.accept()

    def launch_neural_network_visualization(self):
        subprocess.Popen(['python', 'neural_network_visualization.py'])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectorGUI()
    window.show()
    sys.exit(app.exec_()) 