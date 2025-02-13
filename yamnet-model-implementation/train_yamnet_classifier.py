import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import librosa
from collections import Counter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class GunShotDataset(Dataset):
    def __init__(self, embeddings, labels, augment=False):
        self.embeddings = embeddings
        self.labels = labels
        self.augment = augment
        
    def __len__(self):
        return len(self.embeddings)
        
    def augment_embedding(self, embedding):
        """Apply stronger augmentation to embedding."""
        if not self.augment:
            return embedding
            
        # Convert to numpy for augmentation
        if torch.is_tensor(embedding):
            embedding = embedding.numpy()
            
        # Ensure embedding is the right shape
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(-1)
            
        # Multiple augmentation techniques
        augmented = embedding.copy()
        
        # 1. Random scaling with wider range
        scale = np.random.uniform(0.85, 1.15)
        augmented = augmented * scale
        
        # 2. Add gaussian noise with random intensity
        noise_intensity = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_intensity, augmented.shape)
        augmented = augmented + noise
        
        # 3. Random feature dropout with varying probabilities
        dropout_prob = np.random.uniform(0.1, 0.2)
        mask = np.random.binomial(1, 1-dropout_prob, augmented.shape)
        augmented = augmented * mask
        
        # 4. Random feature emphasis
        if np.random.random() < 0.3:
            emphasis = np.random.uniform(1.05, 1.2)
            feature_idx = np.random.randint(0, len(augmented), size=int(len(augmented)*0.2))
            augmented[feature_idx] *= emphasis
            
        # 5. Random smoothing
        if np.random.random() < 0.3:
            window_size = np.random.randint(3, 7)
            kernel = np.ones(window_size) / window_size
            augmented = np.convolve(augmented, kernel, mode='same')
        
        # Normalize
        augmented = (augmented - np.mean(augmented)) / (np.std(augmented) + 1e-6)
        
        # Convert back to tensor
        augmented = torch.FloatTensor(augmented)
        return augmented
        
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        if self.augment and np.random.random() < 0.7:  # 70% chance of augmentation
            embedding = self.augment_embedding(embedding)
        
        if not torch.is_tensor(embedding):
            embedding = torch.FloatTensor(embedding)
        return embedding, self.labels[idx]

class EnhancedGunClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()
        
        self.feature_weights = nn.Parameter(torch.FloatTensor([0.4, 0.3, 0.2, 0.1]))
        self.softmax = nn.Softmax(dim=0)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(768),
            nn.Dropout(dropout_rate),
            
            nn.Linear(768, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

def compute_class_weights(labels):
    class_counts = Counter(labels)
    total = len(labels)
    weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
    return weights

def load_dataset(embeddings_dir):
    """Load and process YAMNet embeddings for each audio file."""
    embeddings = []
    labels = []
    
    for class_name in os.listdir(embeddings_dir):
        class_dir = os.path.join(embeddings_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Process each audio file's embeddings
        file_embeddings = {}
        embedding_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        for emb_file in embedding_files:
            audio_file = emb_file.split('_segment')[0]
            emb_path = os.path.join(class_dir, emb_file)
            embedding = np.load(emb_path)
            
            if audio_file not in file_embeddings:
                file_embeddings[audio_file] = []
            file_embeddings[audio_file].append(embedding)
        
        # Process each audio file
        for audio_file, file_embs in file_embeddings.items():
            # Convert to numpy array for processing
            file_embs = np.array(file_embs)
            
            # Calculate statistics over time segments
            mean_emb = np.mean(file_embs, axis=0)
            std_emb = np.std(file_embs, axis=0)
            max_emb = np.max(file_embs, axis=0)
            min_emb = np.min(file_embs, axis=0)
            
            # Combine features with more emphasis on discriminative statistics
            combined_emb = np.concatenate([
                mean_emb[:256],  # Mean features
                std_emb[256:512],  # Std features
                max_emb[512:768],  # Max features
                min_emb[768:]  # Min features
            ])
            
            # Additional normalization
            combined_emb = (combined_emb - np.mean(combined_emb)) / (np.std(combined_emb) + 1e-6)
            
            embeddings.append(combined_emb)
            labels.append(class_name)
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Global normalization
    embeddings = (embeddings - np.mean(embeddings, axis=0)) / (np.std(embeddings, axis=0) + 1e-6)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    return embeddings, encoded_labels, label_encoder

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, weight_decay=0.001):
    best_combined_acc = 0.0
    best_val_acc = 0.0
    best_model_state = None
    all_metrics = []
    
    # Dynamic weighting based on validation performance trend
    val_weight = 0.7
    window_size = 5
    val_history = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for embeddings, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            # L2 regularization
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += weight_decay * l2_reg
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Update validation weight dynamically
        val_history.append(val_acc)
        if len(val_history) > window_size:
            val_history.pop(0)
            recent_trend = np.mean(np.diff(val_history))
            if recent_trend < 0:  # Validation performance declining
                val_weight = min(0.8, val_weight + 0.02)  # Increase validation importance
            else:
                val_weight = max(0.6, val_weight - 0.01)  # Decrease validation importance
        
        # Learning rate scheduling
        scheduler.step()
        
        # Calculate combined accuracy with dynamic weighting
        combined_acc = ((1 - val_weight) * train_acc + val_weight * val_acc)
        
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss/len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss/len(val_loader),
            'val_acc': val_acc,
            'combined_acc': combined_acc,
            'val_weight': val_weight
        }
        all_metrics.append(metrics)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {metrics['train_loss']:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {metrics['val_loss']:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Combined Acc: {combined_acc:.2f}% (Val Weight: {val_weight:.2f})")
        
        if combined_acc > best_combined_acc:
            best_combined_acc = combined_acc
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'combined_acc': combined_acc,
            }
            print(f"New best model with combined accuracy: {combined_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    best_combined = max(all_metrics, key=lambda x: x['combined_acc'])
    best_val = max(all_metrics, key=lambda x: x['val_acc'])
    best_train = max(all_metrics, key=lambda x: x['train_acc'])
    
    print("\nTraining Summary:")
    print(f"Best Combined Accuracy: {best_combined['combined_acc']:.2f}% "
          f"(Epoch {best_combined['epoch']}, "
          f"Train: {best_combined['train_acc']:.2f}%, "
          f"Val: {best_combined['val_acc']:.2f}%)")
    print(f"Best Validation Accuracy: {best_val['val_acc']:.2f}% "
          f"(Epoch {best_val['epoch']}, "
          f"Train: {best_val['train_acc']:.2f}%)")
    print(f"Best Training Accuracy: {best_train['train_acc']:.2f}% "
          f"(Epoch {best_train['epoch']}, "
          f"Val: {best_train['val_acc']:.2f}%)")
    
    torch.save(best_model_state, 'yamnet_gunshot_classifier.pth')
    np.save('training_metrics.npy', all_metrics)
    
    return best_combined_acc, best_val_acc

def main():
    BATCH_SIZE = 8
    NUM_EPOCHS = 300
    LEARNING_RATE = 0.001
    EMBEDDINGS_DIR = "yamnet_embeddings"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nLoading dataset...")
    embeddings, labels, label_encoder = load_dataset(EMBEDDINGS_DIR)
    np.save('yamnet_label_encoder_classes.npy', label_encoder.classes_)
    
    # Compute class weights
    class_weights = compute_class_weights(labels)
    class_weight_tensor = torch.FloatTensor([class_weights[i] for i in range(len(label_encoder.classes_))]).to(device)
    
    # Split dataset with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets with augmentation
    train_dataset = GunShotDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train),
        augment=True
    )
    val_dataset = GunShotDataset(
        torch.FloatTensor(X_val), 
        torch.LongTensor(y_val),
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    # Initialize model with higher dropout
    model = EnhancedGunClassifier(
        num_classes=len(label_encoder.classes_),
        dropout_rate=0.3
    )
    model = model.to(device)
    
    # Loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.001
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # Initial restart period
        T_mult=2,  # Period multiplier after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Train model
    print("\nStarting training...")
    best_combined_acc, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        NUM_EPOCHS, device
    )
    
    print(f"\nTraining complete!")
    print(f"Best Combined Accuracy: {best_combined_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main() 