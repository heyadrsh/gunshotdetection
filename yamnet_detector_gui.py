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
from urllib.request import urlopen
from threading import Thread
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QProgressBar, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor, QPalette

class GunShotDetector(QThread):
    detection_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        self.window_duration = 2
        self.audio_queue = queue.Queue(maxsize=32)  # Limit queue size
        self.running = False
        self.window_samples = int(self.sample_rate * self.window_duration)
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        
        # Detection thresholds
        self.confidence_threshold = 10.0  # Minimum confidence to consider
        
        # Keywords for detection
        self.primary_keywords = {
            'gunshot': ['gunshot', 'gunfire'],
            'machine_gun': ['machine gun'],
            'cap_gun': ['cap gun']
        }
        
        self.related_keywords = {
            'explosion': ['explosion', 'burst', 'pop'],
            'impact': ['clang', 'bang', 'thud'],
            'mechanical': ['rifle', 'firearm', 'weapon']
        }
        
        # Load YAMNet model
        print("Loading YAMNet model...")
        try:
            tf.get_logger().setLevel('ERROR')
            model_handle = 'https://tfhub.dev/google/yamnet/1'
            self.yamnet_model = hub.load(model_handle)
            
            # Load class names
            class_map_path = urlopen("https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv")
            self.class_names = []
            csv_text = io.TextIOWrapper(class_map_path, encoding='utf-8')
            reader = csv.DictReader(csv_text)
            for row in reader:
                self.class_names.append(row['display_name'])
                
            print("YAMNet model loaded successfully.")
        except Exception as e:
            print(f"Error loading YAMNet model: {e}")
            raise

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Error in audio recording: {status}")
            self.error_signal.emit(f"Audio Error: {status}")
        if self.running:
            try:
                # Only add to queue if not full
                if not self.audio_queue.full():
                    self.audio_queue.put_nowait(indata.copy())
            except queue.Full:
                print("Audio buffer full, dropping frame")

    def process_audio(self, audio):
        try:
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            # Ensure correct length
            if len(audio) < self.window_samples:
                # Pad with zeros if too short
                audio = np.pad(audio, (0, self.window_samples - len(audio)))
            elif len(audio) > self.window_samples:
                # Trim if too long
                audio = audio[:self.window_samples]
            return librosa.util.normalize(audio)
        except Exception as e:
            print(f"Error processing audio: {e}")
            self.error_signal.emit(f"Processing Error: {e}")
            return None

    def check_keywords(self, class_name, confidence):
        # Check primary keywords
        for category, keywords in self.primary_keywords.items():
            if any(keyword in class_name.lower() for keyword in keywords):
                return True, category, confidence
        
        # Check related keywords
        for category, keywords in self.related_keywords.items():
            if any(keyword in class_name.lower() for keyword in keywords):
                return True, category, confidence
        
        return False, None, 0

    def run(self):
        self.running = True
        
        try:
            with sd.InputStream(channels=1,
                              samplerate=self.sample_rate,
                              callback=self.audio_callback,
                              blocksize=int(self.sample_rate * 0.1),
                              device=None,  # Use default device
                              latency='low'):  # Use low latency
                print("Audio stream started")
                while self.running:
                    try:
                        if not self.audio_queue.empty():
                            audio_data = self.audio_queue.get_nowait()
                            
                            # Update buffer with new data
                            self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
                            self.audio_buffer[-len(audio_data):] = audio_data.flatten()
                            
                            # Process audio
                            processed_audio = self.process_audio(self.audio_buffer)
                            if processed_audio is not None:
                                # Get YAMNet predictions
                                scores, embeddings, spectrogram = self.yamnet_model(processed_audio)
                                class_scores = scores.numpy().mean(axis=0)
                                
                                # Check for detections
                                detections = {}
                                for idx, score in enumerate(class_scores):
                                    confidence = score * 100
                                    if confidence > self.confidence_threshold:
                                        is_relevant, category, conf = self.check_keywords(self.class_names[idx], confidence)
                                        if is_relevant:
                                            if category not in detections or confidence > detections[category]['confidence']:
                                                detections[category] = {
                                                    'label': self.class_names[idx],
                                                    'confidence': confidence
                                                }
                                
                                if detections:
                                    self.detection_signal.emit(detections)
                            
                        time.sleep(0.01)  # Short sleep to prevent CPU overload
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"Error in processing loop: {e}")
                        self.error_signal.emit(f"Processing Error: {e}")
                        time.sleep(0.1)  # Sleep longer on error
                        
        except Exception as e:
            print(f"Error in audio stream: {e}")
            self.error_signal.emit(f"Stream Error: {e}")
        finally:
            self.running = False
            print("Audio stream stopped")

    def stop(self):
        self.running = False

class DetectionWidget(QFrame):
    def __init__(self, category, parent=None):
        super().__init__(parent)
        self.category = category
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        self.title = QLabel(self.category.replace('_', ' ').title())
        self.title.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        layout.addWidget(self.title)
        
        # Confidence bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setTextVisible(True)
        layout.addWidget(self.confidence_bar)
        
        # Style
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setMidLineWidth(1)
        
    def update_detection(self, detection_info):
        confidence = detection_info['confidence']
        self.confidence_bar.setValue(int(confidence))
        self.confidence_bar.setFormat(f"{detection_info['label']}: {confidence:.1f}%")
        
        # Color based on confidence
        if confidence > 40:
            color = "red"
        elif confidence > 20:
            color = "orange"
        else:
            color = "yellow"
            
        self.confidence_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_detector()
        
    def setup_ui(self):
        self.setWindowTitle("Gunshot Detection System")
        self.setMinimumSize(600, 400)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Title
        title = QLabel("Real-time Gunshot Detection")
        title.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Status label
        self.status_label = QLabel("Monitoring for gunshots...")
        self.status_label.setFont(QFont('Arial', 12))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Detection widgets
        self.detection_widgets = {}
        
        # Primary detections
        primary_group = QFrame()
        primary_group.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        primary_layout = QVBoxLayout(primary_group)
        primary_layout.addWidget(QLabel("Primary Detections"))
        
        primary_detectors = QHBoxLayout()
        for category in ['gunshot', 'machine_gun', 'cap_gun']:
            widget = DetectionWidget(category)
            self.detection_widgets[category] = widget
            primary_detectors.addWidget(widget)
        primary_layout.addLayout(primary_detectors)
        layout.addWidget(primary_group)
        
        # Related detections
        related_group = QFrame()
        related_group.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        related_layout = QVBoxLayout(related_group)
        related_layout.addWidget(QLabel("Related Detections"))
        
        related_detectors = QHBoxLayout()
        for category in ['explosion', 'impact', 'mechanical']:
            widget = DetectionWidget(category)
            self.detection_widgets[category] = widget
            related_detectors.addWidget(widget)
        related_layout.addLayout(related_detectors)
        layout.addWidget(related_group)
        
        # Alert label
        self.alert_label = QLabel("")
        self.alert_label.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        self.alert_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.alert_label)
        
        # Add error label
        self.error_label = QLabel("")
        self.error_label.setFont(QFont('Arial', 10))
        self.error_label.setStyleSheet("color: orange;")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.error_label)
        
    def setup_detector(self):
        self.detector = GunShotDetector()
        self.detector.detection_signal.connect(self.handle_detection)
        self.detector.error_signal.connect(self.handle_error)
        self.detector.start()
        
    def handle_detection(self, detections):
        any_primary = False
        total_confidence = 0
        detection_count = 0
        
        # Update detection widgets
        for category, detection in detections.items():
            if category in self.detection_widgets:
                self.detection_widgets[category].update_detection(detection)
                if category in ['gunshot', 'machine_gun', 'cap_gun']:
                    any_primary = True
                    total_confidence += detection['confidence']
                    detection_count += 1
        
        # Update alert status
        if any_primary:
            avg_confidence = total_confidence / detection_count
            self.alert_label.setText("⚠️ GUNSHOT DETECTED ⚠️")
            self.alert_label.setStyleSheet("color: red;")
            self.status_label.setText(f"Alert! Gunshot detected with {avg_confidence:.1f}% average confidence")
            self.status_label.setStyleSheet("color: red;")
        else:
            self.alert_label.setText("")
            self.status_label.setText("Monitoring for gunshots...")
            self.status_label.setStyleSheet("")
            
    def handle_error(self, error_msg):
        self.error_label.setText(error_msg)
        # Clear error message after 5 seconds
        QTimer.singleShot(5000, lambda: self.error_label.setText(""))
        
    def closeEvent(self, event):
        self.detector.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 