import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("BINARY EYE DISEASE CLASSIFICATION - PyTorch")
print("Normal vs Abnormal")
print("=" * 70)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"\n✓ Using device: {DEVICE}")

# Custom Dataset
class EyeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load data
print("\nLoading dataset...")
dataset_dir = Path("dataset")

normal_imgs = list((dataset_dir / "Normal").glob("*.[jp][pn][g]"))
abnormal_imgs = list((dataset_dir / "Abnormal").glob("*.[jp][pn][g]"))

all_images = normal_imgs + abnormal_imgs
all_labels = [0] * len(normal_imgs) + [1] * len(abnormal_imgs)

print(f"✓ Normal: {len(normal_imgs)} images")
print(f"✓ Abnormal: {len(abnormal_imgs)} images")
print(f"✓ Total: {len(all_images)} images")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"\n✓ Training: {len(X_train)} images")
print(f"✓ Validation: {len(X_val)} images")
print(f"✓ Test: {len(X_test)} images")

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Simple transform for validation/test
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = EyeDataset(X_train, y_train, train_transform)
val_dataset = EyeDataset(X_val, y_val, val_transform)
test_dataset = EyeDataset(X_test, y_test, val_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Calculate class weights
class_counts = [len(normal_imgs), len(abnormal_imgs)]
total = sum(class_counts)
class_weights = torch.tensor([total/class_counts[0], total/class_counts[1]], 
                              dtype=torch.float).to(DEVICE)
print(f"\n✓ Class weights: Normal={class_weights[0]:.3f}, Abnormal={class_weights[1]:.3f}")

# Build model
print("\nBuilding model...")
model = models.resnet34(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

print(f"✓ Model: ResNet34")
print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training function
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# Validation function
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# Training loop
print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    scheduler.step(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    print()

# Load best model
model.load_state_dict(torch.load('models/best_model.pth'))

# Final evaluation on test set
print("\n" + "=" * 70)
print("FINAL EVALUATION ON TEST SET")
print("=" * 70)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calculate metrics
test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n✓ Test Accuracy: {test_acc:.2f}%")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, 
                          target_names=['Normal', 'Abnormal'], 
                          digits=3))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Abnormal'],
            yticklabels=['Normal', 'Abnormal'])
plt.title(f'Binary Eye Disease Classification\nTest Accuracy: {test_acc:.1f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_binary.png', dpi=300)
print("\n✓ Confusion matrix saved: confusion_matrix_binary.png")

print("\n" + "=" * 70)
print("✓ TRAINING COMPLETE!")
print("=" * 70)
print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
print(f"Final test accuracy: {test_acc:.2f}%")
print(f"\nModel saved: models/best_model.pth")
