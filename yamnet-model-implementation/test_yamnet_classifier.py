import os
import numpy as np
import torch
import librosa
from train_yamnet_classifier import YAMNetGunClassifier
from yamnet_feature_extractor import YAMNetExtractor

class GunShotPredictor:
    def __init__(self, model_path='yamnet_gunshot_classifier.pth'):
        # Load label encoder classes
        self.label_encoder_classes = np.load('yamnet_label_encoder_classes.npy')
        
        # Initialize YAMNet feature extractor
        self.feature_extractor = YAMNetExtractor()
        
        # Load trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YAMNetGunClassifier(num_classes=len(self.label_encoder_classes))
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully. Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    def predict(self, audio_path):
        """Predict gun type from audio file."""
        try:
            # Load and process audio
            audio = self.feature_extractor.load_and_process_audio(audio_path)
            if audio is None:
                return None, 0, []
            
            # Extract YAMNet embeddings
            embedding = self.feature_extractor.extract_embeddings(audio)
            if embedding is None:
                return None, 0, []
            
            # Convert to torch tensor
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(embedding_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                probabilities = probabilities.cpu().numpy() * 100
                
                predicted_class = torch.argmax(outputs, dim=1).item()
                predicted_type = self.label_encoder_classes[predicted_class]
                confidence = probabilities[predicted_class]
                
                return predicted_type, confidence, probabilities
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None, 0, []

def main():
    # Initialize predictor
    predictor = GunShotPredictor()
    
    while True:
        # Get audio file path from user
        audio_path = input("\nEnter path to audio file (or 'q' to quit): ")
        if audio_path.lower() == 'q':
            break
            
        if not os.path.exists(audio_path):
            print("File not found!")
            continue
            
        # Make prediction
        predicted_type, confidence, probabilities = predictor.predict(audio_path)
        
        if predicted_type is None:
            print("Could not process audio file!")
            continue
            
        # Print results
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Predicted Gun Type: {predicted_type}")
        print(f"Confidence: {confidence:.1f}%")
        print("\nAll Probabilities:")
        
        # Sort probabilities in descending order
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices:
            if probabilities[idx] > 1.0:  # Only show probabilities > 1%
                print(f"{predictor.label_encoder_classes[idx]}: {probabilities[idx]:.1f}%")
        
        print("-" * 50)

if __name__ == "__main__":
    main() 