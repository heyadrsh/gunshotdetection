import sys
import os
import numpy as np
import torch
import librosa
import time
from functools import lru_cache
from train_model_colab import GunShotModel
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Optimized audio preprocessing parameters
SAMPLE_RATE = 22050
MAX_DURATION = 5.0  # Reduced from 10.0 to speed up processing
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

@lru_cache(maxsize=32)
def load_and_preprocess_audio(file_path, augment=False):
    """Optimized audio preprocessing with caching."""
    try:
        # Load audio with optimized parameters
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_DURATION, 
                              res_type='kaiser_fast', mono=True)  # Using faster resampling
        
        # Ensure consistent length through efficient numpy operations
        target_length = int(MAX_DURATION * SAMPLE_RATE)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # Compute mel spectrogram with optimized parameters
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            power=2.0,
            center=False  # Faster processing
        )
        
        # Efficient log scaling and normalization
        mel_spec_db = librosa.power_to_db(mel_spec + 1e-9, ref=np.max)
        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db) + 1e-9
        mel_spec_db = (mel_spec_db - mean) / std
        
        return mel_spec_db.astype(np.float32)  # Use float32 for better memory efficiency
    except Exception as e:
        print_colored(f"Error processing {file_path}: {str(e)}", Fore.RED)
        return None

def print_colored(text, color=Fore.WHITE, style=Style.NORMAL):
    """Print colored text using colorama."""
    print(f"{style}{color}{text}{Style.RESET_ALL}")

def load_model(model_path, num_classes):
    """Load the trained model with optimization."""
    try:
        print_colored("\n📂 Loading model...", Fore.CYAN, Style.BRIGHT)
        device = torch.device('cpu')  # Force CPU
        print_colored(f"🖥️  Using device: {device}", Fore.CYAN)
        
        # Create model instance
        model = GunShotModel(num_classes=num_classes)
        
        # Load checkpoint with optimization
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print_colored("✅ Model loaded successfully!", Fore.GREEN)
        
        model = model.to(device)
        model.eval()
        
        # Enable torch inference optimizations
        torch.set_grad_enabled(False)
        
        return model, device
    except Exception as e:
        print_colored(f"❌ Error loading model: {str(e)}", Fore.RED)
        sys.exit(1)

def predict_gunshot(model, audio_path, label_encoder_classes, device):
    """
    Predict the type of gunshot from an audio file
    """
    print_colored("Processing audio file...", Fore.YELLOW)
    start_time = time.time()
    
    try:
        # Process audio file
        features = load_and_preprocess_audio(audio_path)
        if features is None:
            return None, 0, []
        
        preprocessing_time = time.time() - start_time
        print_colored(f"Preprocessing completed in {preprocessing_time:.2f} seconds", Fore.YELLOW)
        
        # Convert to tensor and prepare for model
        features = torch.from_numpy(features).float().unsqueeze(0).to(device)
        
        # Make prediction
        print_colored("Running inference...", Fore.YELLOW)
        inference_start = time.time()
        with torch.inference_mode():
            outputs = model(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            probabilities = probabilities.cpu().numpy() * 100
            
            # Get predicted class
            predicted_class = torch.argmax(outputs, dim=1).item()
            predicted_type = label_encoder_classes[predicted_class]
            confidence = probabilities[predicted_class]
        
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        print_colored(f"Inference completed in {inference_time:.2f} seconds", Fore.YELLOW)
        print_colored(f"Total processing time: {total_time:.2f} seconds", Fore.YELLOW)
        
        return predicted_type, confidence, probabilities
        
    except Exception as e:
        print_colored(f"Error in prediction: {str(e)}", Fore.RED)
        return None, 0, []

def validate_file(file_path):
    """Validate the input file."""
    if not os.path.exists(file_path):
        print_colored(f"❌ Error: File '{file_path}' does not exist.", Fore.RED)
        return False
    
    if not file_path.lower().endswith('.wav'):
        print_colored("❌ Error: Please provide a WAV file.", Fore.RED)
        return False
    
    return True

def main():
    try:
        # Print system information
        print_colored("🚀 Gunshot Detection System (CPU Optimized)", Fore.CYAN, Style.BRIGHT)
        print_colored(f"Python version: {sys.version}", Fore.CYAN)
        
        # Load label encoder classes
        print_colored("\n📊 Loading class labels...", Fore.CYAN)
        label_encoder_classes = np.load('label_encoder_classes.npy')
        print_colored(f"Found {len(label_encoder_classes)} classes:", Fore.GREEN)
        for idx, gun_type in enumerate(label_encoder_classes, 1):
            print_colored(f"  {idx}. {gun_type}", Fore.GREEN)

        # Load model
        model, device = load_model('best_model.pth', len(label_encoder_classes))

        while True:
            print_colored("\n" + "="*50, Fore.YELLOW)
            print_colored("Enter 'q' to quit or provide a WAV file path:", Fore.YELLOW)
            user_input = input(f"{Fore.GREEN}> {Style.RESET_ALL}").strip()
            
            if user_input.lower() == 'q':
                print_colored("\n👋 Thank you for using Gunshot Detection System!", Fore.CYAN)
                break
            
            if not validate_file(user_input):
                continue
            
            # Process the file
            predicted_type, confidence, probabilities = predict_gunshot(
                model, user_input, label_encoder_classes, device
            )
            
            if predicted_type is None:
                continue
            
            # Display results
            print_colored("\n🎯 Results", Fore.CYAN, Style.BRIGHT)
            print_colored(f"Predicted Gun Type: {predicted_type}", Fore.GREEN, Style.BRIGHT)
            print_colored(f"Confidence: {confidence:.2f}%", Fore.GREEN)
            
            print_colored("\n📊 Probabilities for all classes:", Fore.CYAN)
            # Sort probabilities in descending order
            sorted_indices = np.argsort(probabilities)[::-1]
            for idx in sorted_indices:
                gun_type = label_encoder_classes[idx]
                prob = probabilities[idx]
                color = Fore.GREEN if prob > 50 else Fore.YELLOW if prob > 20 else Fore.RED
                print_colored(f"{gun_type}: {prob:.2f}%", color)

    except KeyboardInterrupt:
        print_colored("\n\n👋 Program interrupted by user", Fore.YELLOW)
    except Exception as e:
        print_colored(f"\n❌ An unexpected error occurred: {str(e)}", Fore.RED)
        import traceback
        print_colored(traceback.format_exc(), Fore.RED)

if __name__ == '__main__':
    main() 