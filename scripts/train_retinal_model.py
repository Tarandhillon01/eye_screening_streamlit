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
print("MODE 2: RETINAL DISEASE CLASSIFICATION - PyTorch")
print("4 Classes: Normal, Cataract, Diabetic Retinopathy, Glaucoma")
print("=" * 70)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"\n✓ Using device: {DEVICE}")

# Custom Dataset
class RetinalDataset(Dataset):
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
print("\nLoading retinal dataset...")
dataset_dir = Path("retinal_dataset")

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
all_images = []
all_labels = []

for idx, class_name in enumerate(class_names):
    class_path = dataset_dir / class_name
    images = list(class_path.glob("*.[jp][pn][g]"))
    all_images.extend(images)
    all_labels.extend([idx] * len(images))
    print(f"✓ {class_name}: {len(images)} images")

print(f"\n✓ Total: {len(all_images)} images")
print(f"✓ Classes: {class_names}")

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

# Data augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = RetinalDataset(X_train, y_train, train_transform)
val_dataset = RetinalDataset(X_val, y_val, val_transform)
test_dataset = RetinalDataset(X_test, y_test, val_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Build model
print("\nBuilding model...")
model = models.resnet34(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
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
print("TRAINING MODE 2")
print("=" * 70)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    scheduler.step(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/retinal_model.pth')
        print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    print()

# Load best model
model.load_state_dict(torch.load('models/retinal_model.pth'))

# Final evaluation
print("\n" + "=" * 70)
print("FINAL EVALUATION ON TEST SET")
print("=" * 70)

model.eval()
all_preds = []
all_labels_test = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels_test.extend(labels.numpy())

# Calculate metrics
test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels_test))
print(f"\n✓ Test Accuracy: {test_acc:.2f}%")

print("\nClassification Report:")
print(classification_report(all_labels_test, all_preds, 
                          target_names=['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal'],
                          labels=[0, 1, 2, 3],
                          digits=3,
                          zero_division=0))

# Confusion Matrix
cm = confusion_matrix(all_labels_test, all_preds, labels=[0, 1, 2, 3])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cataract', 'Diabetic\nRetinopathy', 'Glaucoma', 'Normal'],
            yticklabels=['Cataract', 'Diabetic\nRetinopathy', 'Glaucoma', 'Normal'])
plt.title(f'Mode 2: Retinal Disease Classification\nTest Accuracy: {test_acc:.1f}%', 
          fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_retinal.png', dpi=300)
print("\n✓ Confusion matrix saved: confusion_matrix_retinal.png")

# Save class names
with open('models/retinal_classes.txt', 'w') as f:
    f.write('\n'.join(['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']))

print("\n" + "=" * 70)
print("✓ MODE 2 TRAINING COMPLETE!")
print("=" * 70)
print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
print(f"Final test accuracy: {test_acc:.2f}%")
print(f"\nModel saved: models/retinal_model.pth")
print(f"Classes saved: models/retinal_classes.txt")
