import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import colorednoise as cn
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns

def generate_colored_noise(length, noise_type='white'):
    if noise_type == 'white':
        return np.random.normal(0, 1, length)
    elif noise_type == 'brown':
        return cn.powerlaw_psd_gaussian(2, length)
    elif noise_type == 'pink':
        return cn.powerlaw_psd_gaussian(1, length)
    return np.random.normal(0, 1, length)

def apply_augmentation(audio, sr):
    """Apply various augmentations to the audio."""
    # Store original length
    orig_len = len(audio)
    
    # Time stretching
    if random.random() < 0.5:
        stretch_factor = np.random.uniform(0.8, 1.2)
        audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    # Pitch shifting
    if random.random() < 0.5:
        n_steps = np.random.randint(-4, 4)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    # Add colored noise
    if random.random() < 0.5:
        noise_types = ['white', 'pink', 'brown']
        noise_type = random.choice(noise_types)
        noise = generate_colored_noise(len(audio), noise_type)
        noise_factor = np.random.uniform(0.001, 0.015)
        audio = audio + noise_factor * noise
    
    # Random volume change
    if random.random() < 0.5:
        audio = audio * np.random.uniform(0.8, 1.2)
    
    # Ensure output length matches input length
    if len(audio) > orig_len:
        audio = audio[:orig_len]
    elif len(audio) < orig_len:
        audio = np.pad(audio, (0, orig_len - len(audio)))
    
    return audio

class AudioDataset(Dataset):
    def __init__(self, X, y, is_train=False):
        self.X = X
        self.y = y
        self.is_train = is_train
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.is_train and random.random() < 0.5:
            # Add random noise during training for robustness
            noise = np.random.normal(0, 0.001, x.shape)
            x = x + noise
        return torch.FloatTensor(x), torch.LongTensor([self.y[idx]])[0]

def load_and_preprocess_audio(file_path, max_duration=10.0, sr=22050, augment=False):
    try:
        audio, sr = librosa.load(file_path, sr=sr, duration=max_duration)
        
        # Apply augmentation during training
        if augment:
            audio = apply_augmentation(audio, sr)
        
        # Ensure consistent length
        target_length = int(max_duration * sr)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Extract features with more mel bands
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            power=2.0
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

class GunShotModel(nn.Module):
    def __init__(self, num_classes):
        super(GunShotModel, self).__init__()
        self.num_classes = num_classes
        
        # Enhanced CNN architecture
        self.features = nn.Sequential(
            # First block with residual connection
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # Second block with increased channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # Third block with attention
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            
            # Fourth block with increased complexity
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x

def prepare_dataset(data_dir):
    X = []
    y = []
    label_encoder = LabelEncoder()
    
    # Define allowed gun types
    allowed_guns = ['AK-12', 'AK-47', 'M249', 'Zastava M92', 'MP5', 'MG-42']
    
    gun_types = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d in allowed_guns]
    print(f"Found gun types: {gun_types}")
    
    if not gun_types:
        raise ValueError("No gun type directories found in the dataset")
    
    # Process each gun type with augmentation
    for gun_type in gun_types:
        gun_dir = os.path.join(data_dir, gun_type)
        files = [f for f in os.listdir(gun_dir) if f.endswith('.wav')]
        
        print(f"Processing {len(files)} files for {gun_type}")
        
        # Adjust augmentation based on gun type
        n_augment = 2  # default augmentation
        if gun_type in ['MP5', 'MG-42']:  # More augmentation for moderate performers
            n_augment = 3
        
        for file in tqdm(files, desc=f"Processing {gun_type}"):
            file_path = os.path.join(gun_dir, file)
            
            # Original sample
            features = load_and_preprocess_audio(file_path, augment=False)
            if features is not None:
                X.append(features)
                y.append(gun_type)
            
            # Augmented samples with more variations for challenging classes
            for _ in range(n_augment):
                features = load_and_preprocess_audio(file_path, augment=True)
                if features is not None:
                    X.append(features)
                    y.append(gun_type)
    
    if not X:
        raise ValueError("No valid audio files were processed")
    
    y = label_encoder.fit_transform(y)
    X = np.array(X)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, label_encoder

def train_model(model, train_loader, val_loader, criterion, optimizer, label_encoder, num_epochs=50, device='cuda'):
    best_val_acc = 0
    train_losses = []
    val_losses = []
    patience = 15
    patience_counter = 0
    num_classes = len(label_encoder.classes_)
    
    # Metrics history
    precision_history = []
    recall_history = []
    f1_history = []
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Training metrics
        train_predictions = []
        train_targets = []
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Collect predictions and targets for metrics
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        train_acc = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        
        # Calculate training metrics
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_targets, train_predictions, average='weighted'
        )
        
        # Training confusion matrix
        train_cm = confusion_matrix(train_targets, train_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f'Training Confusion Matrix - Epoch {epoch+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'train_confusion_matrix_epoch_{epoch+1}.png')
        plt.close()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_predictions = []
        val_targets = []
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Collect predictions and targets for metrics
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
                # Per-class accuracy
                c = predicted.eq(targets)
                for i in range(len(targets)):
                    label = targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        val_acc = 100. * correct / total
        val_losses.append(val_loss / len(val_loader))
        
        # Calculate validation metrics
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_targets, val_predictions, average='weighted'
        )
        
        # Store metrics history
        precision_history.append(val_precision)
        recall_history.append(val_recall)
        f1_history.append(val_f1)
        
        # Validation confusion matrix
        val_cm = confusion_matrix(val_targets, val_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f'Validation Confusion Matrix - Epoch {epoch+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'val_confusion_matrix_epoch_{epoch+1}.png')
        plt.close()
        
        # Print detailed metrics
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Training Metrics:')
        print(f'Loss: {train_losses[-1]:.3f} | Accuracy: {train_acc:.2f}%')
        print(f'Precision: {train_precision:.3f} | Recall: {train_recall:.3f} | F1: {train_f1:.3f}')
        
        print(f'\nValidation Metrics:')
        print(f'Loss: {val_losses[-1]:.3f} | Accuracy: {val_acc:.2f}%')
        print(f'Precision: {val_precision:.3f} | Recall: {val_recall:.3f} | F1: {val_f1:.3f}')
        
        # Print per-class metrics
        print(f'\nPer-class Validation Metrics:')
        class_report = classification_report(val_targets, val_predictions,
                                          target_names=label_encoder.classes_,
                                          digits=3)
        print(class_report)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'Learning rate: {current_lr:.6f}')
        
        if val_acc > best_val_acc:
            print('Saving best model...')
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'metrics': {
                    'precision': val_precision,
                    'recall': val_recall,
                    'f1': val_f1,
                    'confusion_matrix': val_cm
                }
            }, 'best_model.pth')
            np.save('label_encoder_classes.npy', label_encoder.classes_)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Plot final training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss History')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot([100 * (1 - l/len(train_loader)) for l in train_losses], label='Train Acc')
    plt.plot([100 * (1 - l/len(val_loader)) for l in val_losses], label='Val Acc')
    plt.title('Accuracy History')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(precision_history, label='Precision')
    plt.plot(recall_history, label='Recall')
    plt.plot(f1_history, label='F1-score')
    plt.title('Metrics History')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    print("Preparing dataset...")
    X, y, label_encoder = prepare_dataset('dataset')
    
    # Calculate class weights for balanced training
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to('cuda')
    
    # Split dataset with higher validation ratio for better evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Create data loaders with adjusted batch size
    train_dataset = AudioDataset(X_train, y_train, is_train=True)
    val_dataset = AudioDataset(X_val, y_val, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GunShotModel(num_classes=len(label_encoder.classes_))
    model = model.to(device)
    
    # Training setup with adjusted parameters
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.02)  # Reduced LR, increased weight decay
    
    # Train model with increased epochs
    train_model(model, train_loader, val_loader, criterion, optimizer, label_encoder, num_epochs=150, device=device)

if __name__ == '__main__':
    main() 