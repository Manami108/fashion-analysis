import os
import json
import torch
import numpy as np
import time
import random
from collections import defaultdict, Counter
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier


train_image_dir = ''
train_annotation_dir = ''
val_image_dir = ''
val_annotation_dir = ''
test_image_dir = ''
test_annotation_dir = ''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# determine thenumber of unique categories in the given annotation directory.
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

# custom data class for loading and processing images labels and segmentation masks for the dataset
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

# Loads the annotations from a directory and pairs them with corresponding images.
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

# Balances the dataset by resampling each class to ensure equal representation since the dataset was imbalanced.
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

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])


# Splits the dataset into train and validation sets for cross-validation while maintaining the class distribution in each fold.
num_folds = 5
labels_resampled = [item['label'] for item in resampled_data]
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Define the class name
class_names = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "shorts", "trousers", "skirt", "short sleeve dress",
    "long sleeve dress", "vest dress", "sling dress"
]

# Extracts features from images in a dataset by applying their segmentation masks and flattening the resulting tensors. 
def extract_features_from_dataset(dataset):
    features = []
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample['image']  
        mask = sample['mask']  
        segmented_image = image * mask.expand_as(image)
        feature = segmented_image.flatten().numpy()
        features.append(feature)
        labels.append(sample['label'])
    return np.array(features), np.array(labels)

fold_results = []

# For each fold, print the accuracy and loss. 
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(resampled_data)), labels_resampled)):
    print(f"\n--- Fold {fold+1}/{num_folds} ---")
    train_subset = [resampled_data[i] for i in train_idx]
    val_subset   = [resampled_data[i] for i in val_idx]

    train_dataset = CombinedDeepFashion2Dataset(train_subset, transform=transform)
    val_dataset   = CombinedDeepFashion2Dataset(val_subset, transform=transform)
    
    # Extract features and labels from each dataset
    print("Extracting training features...")
    X_train, y_train = extract_features_from_dataset(train_dataset)
    print("Extracting validation features...")
    X_val, y_val = extract_features_from_dataset(val_dataset)
    
    print(f"Training Random Forest on {X_train.shape[0]} samples with {X_train.shape[1]} features each...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)
    
    train_accuracy = accuracy_score(y_train, y_train_pred) * 100
    val_accuracy = accuracy_score(y_val, y_val_pred) * 100
    
    print(f"Fold {fold+1} -- Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")
    
    # Plot confusion matrix for the validation set
    cm = confusion_matrix(y_val, y_val_pred, labels=range(num_classes))
    with np.errstate(all='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized) * 100
    
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f"Confusion Matrix (%) - Fold {fold+1}")
    plt.show()
    
    fold_results.append({
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy
    })

# Print the overall results.
print("\nOverall Fold Results:")
for idx, result in enumerate(fold_results, 1):
    print(f"Fold {idx} -- Train Accuracy: {result['train_accuracy']:.2f}%, Validation Accuracy: {result['val_accuracy']:.2f}%")
