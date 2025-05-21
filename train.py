import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

class FruitClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FruitClassifier, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = x.view(-1, 512 * 6 * 6)
        x = self.fc(x)
        return x

def load_data(train_dir, val_dir, batch_size=64):
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def check_image_sizes(dataset_path):
        min_size = 300
        optimal_size = 300
        total_images = 0
        too_small = 0
        different_sizes = set()

        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(root, file)
                        with Image.open(img_path) as img:
                            width, height = img.size
                            total_images += 1
                            different_sizes.add((width, height))
                            
                            if width < min_size or height < min_size:
                                too_small += 1
                                print(f"Warning: Image {img_path} is too small: {width}x{height}")
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")

        print(f"\nDataset statistics for {dataset_path}:")
        print(f"Total images: {total_images}")
        print(f"Images smaller than {min_size}x{min_size}: {too_small}")
        print(f"Different image sizes found: {len(different_sizes)}")
        if len(different_sizes) > 1:
            print("Unique sizes:", sorted(different_sizes))
        print(f"All images will be resized to {optimal_size}x{optimal_size}\n")

    print("Checking training images...")
    check_image_sizes(train_dir)
    print("Checking validation images...")
    check_image_sizes(val_dir)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, train_dataset.class_to_idx

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc='Training', unit='batch'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (lam * (predicted == labels_a).sum().float() + 
                   (1 - lam) * (predicted == labels_b).sum().float())

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation', unit='batch'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    return val_loss, val_accuracy

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, model_save_path='best_model.pth'):
    device = torch.device('cpu')
    print(f'Using device: {device}')
    
    model = model.to(device)
    print(f"Model moved to {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_loss = float('inf')
    patience = 15
    counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, model_save_path)
            print(f'Model saved! Validation Loss: {val_loss:.4f}')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                print(f'Best epoch was {best_epoch + 1} with validation loss: {best_val_loss:.4f}')
                break

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'Current learning rate: {current_lr:.6f}')

def save_class_indices(class_to_idx, file_path='class_indices.json'):
    with open(file_path, 'w') as f:
        json.dump(class_to_idx, f)
    print(f'Метки классов сохранены в {file_path}')

if __name__ == "__main__":
    train_dir = 'D:/work/FruitAIv2/data/Training'
    val_dir = 'D:/work/FruitAIv2/data/Test'

    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    model_save_path = 'best_model.pth'
    indices_save_path = 'class_indices.json'

    train_loader, val_loader, class_to_idx = load_data(train_dir, val_dir, batch_size=batch_size)
    
    save_class_indices(class_to_idx, indices_save_path)
    print(f"Сохранены метки классов в {indices_save_path}")

    num_classes = len(class_to_idx)
    model = FruitClassifier(num_classes)

    train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=learning_rate, model_save_path=model_save_path)