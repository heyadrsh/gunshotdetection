import sounddevice as sd
import numpy as np
import torch
import librosa
import time
from test_model import load_model, predict_gunshot
import threading
from queue import Queue
import keyboard
import sys

class MicrophoneDetector:
    def __init__(self):
        # Audio settings
        self.sample_rate = 22050
        self.duration = 3  # Recording duration in seconds
        self.channels = 1
        
        # Load model and classes
        self.label_encoder_classes = np.load('label_encoder_classes.npy')
        self.model, self.device = load_model('best_model.pth', 
                                           len(self.label_encoder_classes))
        
        # Initialize recording queue
        self.audio_queue = Queue()
        self.is_recording = False
        
    def process_audio(self, audio):
        """Process the recorded audio through the model."""
        try:
            # Ensure correct shape and type
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Zero-pad if audio is too short
            target_length = self.sample_rate * self.duration
            if len(audio) < target_length:
                padding = target_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            else:
                audio = audio[:target_length]
            
            # Extract mel spectrogram with parameters matching training
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=128,
                n_fft=2048,
                hop_length=512,
                win_length=1024,
                fmax=8000,
                power=2.0
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec + 1e-9, ref=np.max)
            
            # Normalize using mean and std (matching training)
            mean = np.mean(mel_spec_db)
            std = np.std(mel_spec_db) + 1e-9
            mel_spec_db = (mel_spec_db - mean) / std
            
            return mel_spec_db.astype(np.float32)
            
        except Exception as e:
            print(f"Error in audio processing: {str(e)}")
            return None
    
    def record_callback(self, indata, frames, time, status):
        """Callback for recording audio."""
        if status:
            print(f"Error in recording: {status}")
            return
        
        # Add audio data to queue
        self.audio_queue.put(indata.copy())
    
    def analyze_recording(self):
        """Analyze the recorded audio."""
        try:
            # Get audio from queue
            audio_data = self.audio_queue.get()
            audio = audio_data.flatten()
            
            # Calculate audio level
            audio_rms = np.sqrt(np.mean(np.square(audio)))
            audio_db = 20 * np.log10(audio_rms + 1e-10)
            
            # Print audio level meter
            level = int((audio_db + 60) / 3)  # Scale to 0-20 range
            level = max(0, min(20, level))
            meter = "█" * level + "░" * (20 - level)
            print(f"\nAudio Level: [{meter}] {audio_db:.1f} dB")
            
            # Lower threshold for silence detection
            if audio_db < -55:  # Adjusted threshold
                print("No significant audio detected. Please ensure there's a clear sound.")
                return
            
            # Process audio
            features = self.process_audio(audio)
            if features is None:
                print("Error: Failed to process audio")
                return
            
            # Make prediction
            result = predict_gunshot(self.model, None, self.label_encoder_classes, 
                                   self.device, preprocessed_features=features)
            
            # Extract results
            predicted_type, confidence, probabilities, timings = result
            
            # Apply temperature scaling to soften predictions
            temperature = 1.5
            probabilities = np.exp(np.log(probabilities) / temperature)
            probabilities = probabilities / np.sum(probabilities) * 100  # Convert to percentage
            
            # Update confidence
            max_prob_idx = np.argmax(probabilities)
            predicted_type = self.label_encoder_classes[max_prob_idx]
            confidence = probabilities[max_prob_idx]
            
            # Print results
            print("\n" + "="*50)
            print("Detection Results:")
            
            if confidence < 30:
                print("No clear gunshot type detected. Confidence too low.")
            else:
                print(f"Detected Gun Type: {predicted_type}")
                print(f"Confidence: {confidence:.1f}%")
            
            print("-"*50)
            print("All Probabilities:")
            
            # Sort probabilities in descending order
            sorted_indices = np.argsort(probabilities)[::-1]
            for idx in sorted_indices:
                prob = probabilities[idx]
                if prob > 1:  # Only show probabilities > 1%
                    print(f"{self.label_encoder_classes[idx]}: {prob:.1f}%")
            
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return None
    
    def start_recording(self):
        """Start recording from microphone."""
        try:
            print("\nStarting microphone detection...")
            print("Press 'r' to record a 3-second clip")
            print("Press 'q' to quit")
            print("\nTip: When recording, ensure the sound is loud enough to see the audio level meter fill up!")
            
            while True:
                if keyboard.is_pressed('q'):
                    print("\nExiting...")
                    break
                
                if keyboard.is_pressed('r') and not self.is_recording:
                    self.is_recording = True
                    print("\nRecording for 3 seconds...")
                    
                    # Start recording
                    with sd.InputStream(samplerate=self.sample_rate,
                                     channels=self.channels,
                                     callback=self.record_callback):
                        sd.sleep(int(self.duration * 1000))  # Duration in milliseconds
                    
                    print("Recording complete, analyzing...")
                    self.analyze_recording()
                    self.is_recording = False
                
                time.sleep(0.1)  # Prevent high CPU usage
                
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"\nError: {str(e)}")
            
def main():
    try:
        # Test audio device
        sd.check_output_settings(samplerate=22050, channels=1, dtype=np.float32)
    except sd.PortAudioError:
        print("Error: Could not access the microphone. Please check your audio settings.")
        sys.exit(1)
    
    detector = MicrophoneDetector()
    detector.start_recording()

if __name__ == "__main__":
    main() 