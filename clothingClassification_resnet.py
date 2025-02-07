import os
import json
import torch
import numpy as np
import time
import random
from collections import defaultdict, Counter
from torch import optim
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_image_dir = ''
train_annotation_dir = ''
val_image_dir = ''
val_annotation_dir = ''
test_image_dir = ''
test_annotation_dir = ''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def determine_num_classes(annotation_dir):
 
    categories = set()
    for anno_file in os.listdir(annotation_dir):
        if anno_file.endswith('.json'):
            anno_path = os.path.join(annotation_dir, anno_file)
            try:
                with open(anno_path, 'r') as f:
                    annotation = json.load(f)
            except json.JSONDecodeError:
                continue
            for key in annotation.keys():
                if key in ('source', 'pair_id'):
                    continue
                raw_id = annotation[key].get('category_id')
                if raw_id is not None:
                    categories.add(raw_id)
    return len(categories)  

class CombinedDeepFashion2Dataset(Dataset):
    def __init__(self, data_list, transform=None, target_size=(256, 256)):

        self.data_list = data_list
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        image_path = sample['image_path']
        label = sample['label']
        segmentation = sample['segmentation']
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        mask = Image.new('L', original_size, 0)
        for polygon in segmentation:
            ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
        mask = mask.resize(self.target_size, Image.NEAREST)

        if self.transform:
            image = self.transform(image)
        mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)
        return {'image': image, 'label': label, 'mask': mask}

def load_annotations(image_dir, annotation_dir, num_classes):

    data_list = []
    for anno_file in os.listdir(annotation_dir):
        if not anno_file.endswith('.json'):
            continue
        anno_path = os.path.join(annotation_dir, anno_file)
        try:
            with open(anno_path, 'r') as f:
                annotation = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: JSON decode error in {anno_file}")
            continue

        for key, value in annotation.items():
            if key in ('source', 'pair_id'):
                continue
            if isinstance(value, dict):
                raw_label = value.get('category_id')
                segmentation = value.get('segmentation')
                if raw_label is not None and 1 <= raw_label <= num_classes and segmentation:
                    label = raw_label - 1  
                    image_file = anno_file.replace('.json', '.jpg')
                    image_path = os.path.join(image_dir, image_file)
                    if os.path.exists(image_path):
                        data_list.append({
                            'image_path': image_path,
                            'label': label,
                            'segmentation': segmentation
                        })
                    else:
                        print(f"Warning: Image not found: {image_path}")
            else:
                print(f"Warning: Unexpected format in {anno_file} for key '{key}'")
    return data_list

def resample_to_target(data_list, num_classes, target_count, random_seed=42):

    random.seed(random_seed)
    label_to_items = defaultdict(list)
    for item in data_list:
        label_to_items[item['label']].append(item)
    
    resampled_data = []
    for label in range(num_classes):
        items = label_to_items[label]
        count = len(items)
        if count < target_count:
            extra = random.choices(items, k=target_count - count)
            new_items = items + extra
        else:
            new_items = random.sample(items, target_count)
        resampled_data.extend(new_items)
    
    return resampled_data

num_classes = determine_num_classes(train_annotation_dir) 
print(f"Determined number of classes: {num_classes}")

train_data = load_annotations(train_image_dir, train_annotation_dir, num_classes)
val_data   = load_annotations(val_image_dir, val_annotation_dir, num_classes)
combined_data = train_data + val_data
print(f"Combined data size: {len(combined_data)}")

original_counts = Counter([item['label'] for item in combined_data])
print("Original label distribution:")
for label, count in original_counts.items():
    print(f"Label {label}: {count} images")

target_count = 1985
resampled_data = resample_to_target(combined_data, num_classes, target_count)
resampled_counts = Counter([item['label'] for item in resampled_data])
print("Resampled label distribution (each should be 1985):")
for label in range(num_classes):
    print(f"Label {label}: {resampled_counts[label]} images")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# This pipeline applies various data augmentation techniques to improve the robustness of the model and prevent overfitting.
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
])

transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

num_folds = 5
batch_size = 32
num_epochs = 50
early_stopping_patience = 10

labels_resampled = [item['label'] for item in resampled_data]
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

fold_results = []
fold_histories = []  
class_names = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "shorts", "trousers", "skirt", "short sleeve dress",
    "long sleeve dress", "vest dress", "sling dress"
]

def extract_segmented_regions(images, masks):
    segmented_images = []
    for img, mask in zip(images, masks):
        segmented_images.append(img * mask)
    return torch.stack(segmented_images)

def plot_confusion_matrix(model, dataloader, classes, fold_num):
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            masks  = batch['mask'].to(device)
            segmented_images = extract_segmented_regions(images, masks).to(device)
            outputs = model(segmented_images)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())
    
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(classes)))
    with np.errstate(all='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized) * 100
    
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f"Confusion Matrix (%) - Fold {fold_num}")
    plt.show()

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(resampled_data)), labels_resampled)):
    print(f"\n--- Fold {fold+1}/{num_folds} ---")
    train_subset = [resampled_data[i] for i in train_idx]
    val_subset   = [resampled_data[i] for i in val_idx]
    
    train_dataset_fold = CombinedDeepFashion2Dataset(train_subset, transform=transform_train)
    val_dataset_fold   = CombinedDeepFashion2Dataset(val_subset, transform=transform_val)
    
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader_fold   = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False, num_workers=4)

# Model is called here. The final fully connected layer is replaced with a custom layer for the number of classes.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, batch in enumerate(train_loader_fold):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            masks  = batch['mask'].to(device)
            
            segmented_images = extract_segmented_regions(images, masks)
            optimizer.zero_grad()
            outputs = model(segmented_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f"Fold {fold+1}, Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader_fold)}], Loss: {loss.item():.4f}")
        
        epoch_train_loss = running_loss / len(train_loader_fold)
        epoch_train_accuracy = 100 * correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader_fold:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                masks  = batch['mask'].to(device)
                segmented_images = extract_segmented_regions(images, masks)
                outputs = model(segmented_images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()
        epoch_val_loss = val_loss / len(val_loader_fold)
        epoch_val_accuracy = 100 * correct_val / total_val
        
        scheduler.step(epoch_val_loss)
        epoch_duration = time.time() - start_time
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_accuracy)
        val_accuracies.append(epoch_val_accuracy)
        
        print(f"Fold {fold+1}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%, " \
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%, Time: {epoch_duration:.2f}s")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f'ResNet_best_model_fold{fold+1}.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered for this fold")
                break

    plot_confusion_matrix(model, val_loader_fold, class_names, fold_num=fold+1)

    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
 
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold+1} Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Fold {fold+1} Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    fold_histories.append({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    })
    
    fold_results.append({
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_train_accuracy': train_accuracies[-1],
        'final_val_accuracy': val_accuracies[-1]
    })

for idx, result in enumerate(fold_results, 1):
    print(f"Fold {idx} -- Train Loss: {result['final_train_loss']:.4f}, Val Loss: {result['final_val_loss']:.4f}, " \
          f"Train Acc: {result['final_train_accuracy']:.2f}%, Val Acc: {result['final_val_accuracy']:.2f}%")
