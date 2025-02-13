import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from tqdm import tqdm

class YAMNetExtractor:
    def __init__(self):
        # Load YAMNet model
        print("Loading YAMNet model...")
        tf.get_logger().setLevel('ERROR')
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("YAMNet model loaded successfully")
        
        # Audio parameters
        self.sample_rate = 16000  # YAMNet requires 16kHz
        
    def load_and_process_audio(self, audio_path):
        """Load and preprocess audio file for YAMNet."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            return audio
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
            
    def extract_embeddings(self, audio):
        """Extract YAMNet embeddings from audio."""
        try:
            # Get YAMNet outputs
            scores, embeddings, mel_spec = self.yamnet_model(audio)
            
            # Average embeddings across time
            embedding = tf.reduce_mean(embeddings, axis=0)
            
            return embedding.numpy()
            
        except Exception as e:
            print(f"Error extracting embeddings: {str(e)}")
            return None
    
    def process_dataset(self, dataset_path, output_path):
        """Process entire dataset and save embeddings."""
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Process each class directory
        for class_name in os.listdir(dataset_path):
            class_dir = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            print(f"\nProcessing class: {class_name}")
            
            # Create output directory for class
            class_output_dir = os.path.join(output_path, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            
            # Process each audio file
            audio_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
            for audio_file in tqdm(audio_files, desc=f"Extracting {class_name}"):
                try:
                    # Load and process audio
                    audio_path = os.path.join(class_dir, audio_file)
                    audio = self.load_and_process_audio(audio_path)
                    
                    if audio is not None:
                        # Extract embeddings
                        embedding = self.extract_embeddings(audio)
                        
                        if embedding is not None:
                            # Save embedding
                            output_file = os.path.join(class_output_dir, 
                                                     f"{os.path.splitext(audio_file)[0]}.npy")
                            np.save(output_file, embedding)
                            
                except Exception as e:
                    print(f"\nError processing {audio_file}: {str(e)}")
                    continue
                    
        print("\nDataset processing complete!") 