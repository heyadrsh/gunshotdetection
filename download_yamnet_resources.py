import os
import tensorflow_hub as hub
import urllib.request
import tensorflow as tf

def download_yamnet_resources():
    print("Downloading YAMNet resources...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Download and save YAMNet model
    print("Downloading YAMNet model...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    model_path = os.path.join('models', 'yamnet_model')
    tf.saved_model.save(model, model_path)
    print("YAMNet model saved successfully!")
    
    # Download class map
    print("Downloading YAMNet class map...")
    class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    class_map_path = os.path.join('models', 'yamnet_class_map.csv')
    urllib.request.urlretrieve(class_map_url, class_map_path)
    print("Class map downloaded successfully!")
    
    print("\nAll YAMNet resources have been downloaded and saved locally!")
    print(f"Model saved to: {model_path}")
    print(f"Class map saved to: {class_map_path}")

if __name__ == "__main__":
    download_yamnet_resources() 