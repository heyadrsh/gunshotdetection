import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import queue
import keyboard
import time
import librosa
import csv
import io
from urllib.request import urlopen
from threading import Thread
from datetime import datetime

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

print("Loading YAMNet model...")
try:
    # Set TensorFlow logging level
    tf.get_logger().setLevel('ERROR')
    
    # Load YAMNet model from local path
    model_path = os.path.join('models', 'yamnet_model')
    if not os.path.exists(model_path):
        raise FileNotFoundError("YAMNet model not found. Please run download_yamnet_resources.py first.")
    yamnet_model = hub.load(model_path)
    
    # Load class names from local file
    class_map_path = os.path.join('models', 'yamnet_class_map.csv')
    if not os.path.exists(class_map_path):
        raise FileNotFoundError("Class map not found. Please run download_yamnet_resources.py first.")
    
    class_names = []
    with open(class_map_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row['display_name'])
            
    print("YAMNet model loaded successfully.")
except Exception as e:
    print(f"Error loading YAMNet model: {e}")
    raise

class RealtimeSoundDetector:
    def __init__(self, sample_rate=16000, window_duration=2):
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.audio_queue = queue.Queue()
        self.running = False
        self.window_samples = int(self.sample_rate * self.window_duration)
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.last_gunshot_time = 0  # To track time between gunshot alerts
        self.alert_cooldown = 2.0  # Minimum seconds between alerts
        
        # Create logs directory if it doesn't exist
        self.logs_dir = "yamnet_logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize log file with timestamp
        self.log_filename = os.path.join(self.logs_dir, f"gunshot_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(self.log_filename, 'w') as f:
            f.write("=== YAMNet Gunshot Detection Log ===\n")
            f.write(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log_detection(self, detections, is_gunshot=False):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_filename, 'a') as f:
            if is_gunshot:
                f.write(f"\n[{timestamp}] !!! GUNSHOT DETECTED !!!\n")
                f.write("Detected sounds that triggered the alert:\n")
            else:
                f.write(f"\n[{timestamp}] Sound Detection:\n")
            for detection in detections:
                f.write(f"- {detection}\n")
            f.write("-" * 50 + "\n")
        
    def process_audio(self, audio):
        try:
            # Ensure audio is mono and has correct sample rate
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            return audio
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            raise

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Error in audio recording: {status}")
        if self.running:
            self.audio_queue.put(indata.copy())

    def check_for_gunshot(self, class_names, scores):
        detected_keywords = []
        for keyword, threshold in GUNSHOT_KEYWORDS.items():
            if keyword in class_names:
                idx = class_names.index(keyword)
                confidence = scores[idx] * 100
                if confidence >= threshold:
                    detected_keywords.append(f"{keyword}: {confidence:.1f}%")
        return detected_keywords

    def analyze_audio(self):
        while self.running:
            try:
                # Get audio data from queue
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    
                    # Roll the buffer and add new data
                    self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
                    self.audio_buffer[-len(audio_data):] = audio_data.flatten()
                    
                    # Process audio
                    processed_audio = self.process_audio(self.audio_buffer)
                    
                    # Get YAMNet predictions
                    scores, embeddings, spectrogram = yamnet_model(tf.convert_to_tensor(processed_audio))
                    scores = scores.numpy()
                    
                    # Get mean scores across frames
                    class_scores = scores.mean(axis=0)
                    
                    # Check for gunshot sounds
                    gunshot_detections = self.check_for_gunshot(class_names, class_scores)
                    
                    # Get current time for cooldown check
                    current_time = time.time()
                    
                    if gunshot_detections and (current_time - self.last_gunshot_time) >= self.alert_cooldown:
                        # Clear the current line and move to a new line for the alert
                        print("\n\033[K", end='')
                        print("\n!!! GUNSHOT DETECTED !!!")
                        print("Detected sounds that triggered the alert:")
                        for detection in gunshot_detections:
                            print(f"- {detection}")
                        print("-" * 50)  # Separator line
                        self.log_detection(gunshot_detections, is_gunshot=True)
                        self.last_gunshot_time = current_time
                    
                    # Get top 5 predicted classes for display
                    top_classes = np.argsort(class_scores)[-5:][::-1]
                    results = []
                    for idx in top_classes:
                        confidence = class_scores[idx] * 100
                        if confidence > 5.0:  # Only show predictions with >5% confidence
                            results.append(f"{class_names[idx]}: {confidence:.1f}%")
                    
                    # Print current detections
                    print("\033[K", end='\r')  # Clear line
                    print(" | ".join(results), end='\r', flush=True)
                    
                    # Log if we have any significant detections
                    if results:
                        self.log_detection(results)
                    
            except Exception as e:
                print(f"\nError analyzing audio: {e}")
            time.sleep(0.01)

    def start(self):
        print("\nGunshot Detection System")
        print("Monitoring for the following sounds:")
        for keyword, threshold in GUNSHOT_KEYWORDS.items():
            print(f"- {keyword} (threshold: {threshold}%)")
        print(f"\nLogging detections to: {self.log_filename}")
        print("Press 'q' to quit\n")
        print("Listening for sounds...")
        
        try:
            self.running = True
            
            # Start analysis thread
            analysis_thread = Thread(target=self.analyze_audio)
            analysis_thread.daemon = True
            analysis_thread.start()
            
            # Start audio stream
            with sd.InputStream(channels=1,
                              samplerate=self.sample_rate,
                              callback=self.audio_callback,
                              blocksize=int(self.sample_rate * 0.1)):  # Process 100ms chunks
                while self.running:
                    if keyboard.is_pressed('q'):
                        print("\nQuitting...")
                        self.running = False
                        break
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.running = False
            time.sleep(0.5)  # Allow time for threads to clean up
            
            # Log session end
            with open(self.log_filename, 'a') as f:
                f.write(f"\nSession ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Session ended by user.\n")
        
        print("\nProgram terminated by user.")
        print(f"Detection log saved to: {self.log_filename}")

if __name__ == "__main__":
    detector = RealtimeSoundDetector()
    detector.start() 