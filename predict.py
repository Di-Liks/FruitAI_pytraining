# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import transforms
import os
import json
from PIL import Image
import numpy as np
import cv2

class FruitClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FruitClassifier, self).__init__()
        # Первый блок свертки
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
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

def load_model_and_indices(model_path='best_model.pth', indices_path='class_indices.json'):
    with open(indices_path, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = FruitClassifier(len(class_to_idx))
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, idx_to_class

def resize_with_padding(image, target_size=(500, 500)):
    width, height = image.size
    
    if width < target_size[0] and height < target_size[1]:
        ratio = max(target_size[0] / width, target_size[1] / height)
        new_size = (int(width * ratio), int(height * ratio))
    else:
        ratio = min(target_size[0] / width, target_size[1] / height)
        new_size = (int(width * ratio), int(height * ratio))
    
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

    new_image = Image.new("RGB", target_size, (0, 0, 0))
    new_image.paste(resized_image, (
        (target_size[0] - new_size[0]) // 2,
        (target_size[1] - new_size[1]) // 2
    ))
    return new_image

def smart_resize(image, min_size=500):
    boundaries = detect_object_boundaries(image)
    if not boundaries:
        return resize_with_padding(image, (min_size, min_size))
    
    x, y, w, h = boundaries
    width, height = image.size
    
    object_area = w * h
    total_area = width * height
    object_ratio = object_area / total_area
    
    if object_ratio < 0.5:
        padding = int(min(w, h) * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        image = image.crop((x, y, x + w, y + h))
    
    return resize_with_padding(image, (min_size, min_size))

def detect_object_boundaries(image):
    img_np = np.array(image)
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_np.shape[1] - x, w + 2 * padding)
    h = min(img_np.shape[0] - y, h + 2 * padding)
    
    return (x, y, w, h)

def process_test_images(model, idx_to_class, test_dir='testing', output_dir='processed_images'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(test_dir):
        print(f"Директория {test_dir} не найдена!")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"В директории {test_dir} не найдено изображений!")
        return

    print("\nРезультаты классификации:")
    print("-" * 50)

    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
            
            resized_image = smart_resize(image, min_size=500)
            input_tensor = transform(resized_image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = idx_to_class[predicted.item()]
                confidence_percent = confidence.item() * 100

            output_filename = f"processed_{image_file}"
            output_path = os.path.join(output_dir, output_filename)
            resized_image.save(output_path)

            print(f"Изображение: {image_file}")
            print(f"Предсказанный класс: {predicted_class}")
            print(f"Уверенность: {confidence_percent:.2f}%")
            print(f"Обработанное изображение сохранено: {output_path}")
            print("-" * 50)

        except Exception as e:
            print(f"Ошибка при обработке {image_file}: {str(e)}")
            print("-" * 50)

if __name__ == "__main__":
    model, idx_to_class = load_model_and_indices()

    process_test_images(model, idx_to_class)