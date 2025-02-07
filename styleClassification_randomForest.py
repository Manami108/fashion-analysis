import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import copy

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

base_dir = ""

class_names = [
    "conservative", "dressy", "ethnic", "fairy", "feminine", "gal", "girlish", 
    "kireime-casual", "lolita", "mode", "natural", "retro", "rock", "street"
]

def is_valid_file(path):
    if os.path.basename(path).startswith("._"):
        return False
    return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))


forest_transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor()         
])

forest_dataset = datasets.ImageFolder(
    root=base_dir,
    transform=forest_transform,
    is_valid_file=is_valid_file
)

print("Class to index mapping:")
print(forest_dataset.class_to_idx)

dataset_size = len(forest_dataset)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size

torch.manual_seed(42)
forest_train_dataset, forest_val_dataset = random_split(forest_dataset, [train_size, val_size])
num_classes = len(forest_dataset.classes)
print(f"\nTotal samples: {dataset_size}, Training samples: {train_size}, Validation samples: {val_size}")
print("Number of classes:", num_classes)

# Converts images into flattened feature vectors for use in the Random Forest classifier.
def extract_features(dataset):
    X = []
    y = []
    for img, label in dataset:
        X.append(img.numpy().flatten())
        y.append(label)
    return np.array(X), np.array(y)

print("\nExtracting features from the training dataset...")
X_train, y_train = extract_features(forest_train_dataset)
print("Feature vector shape (train):", X_train.shape)

print("Extracting features from the validation dataset...")
X_val, y_val = extract_features(forest_val_dataset)
print("Feature vector shape (validation):", X_val.shape)

# Initializes a Random Forest classifier
print("\nTraining the Random Forest classifier...")
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
start_time = time.time()
forest_clf.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Random Forest training time: {training_time:.2f} seconds")

# evaluation: Predict on validation set
y_pred = forest_clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\nRandom Forest Accuracy: {acc:.4f}")

# Compute confusion matrix and classification report
cm = confusion_matrix(y_val, y_pred)
class_labels = [k for k, v in sorted(forest_dataset.class_to_idx.items(), key=lambda item: item[1])]
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=class_labels))

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=True, title="Confusion Matrix (%)", cmap=plt.cm.Blues):

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized) * 100  # convert to percentages
        cm_to_show = cm_normalized
        print("Normalized confusion matrix (in %):")
    else:
        cm_to_show = cm
        print("Confusion matrix, without normalization:")
    print(cm_to_show)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm_to_show, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    fmt = '.0f'
    thresh = cm_to_show.max() / 2.
    for i, j in itertools.product(range(cm_to_show.shape[0]), range(cm_to_show.shape[1])):
        plt.text(j, i, format(cm_to_show[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_to_show[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(cm, classes=class_labels)

print("\n=== Comparison Summary ===")
print("Deep Learning Model (ResNet50) vs Random Forest")
print("- The Random Forest classifier uses raw, flattened pixel values (after resizing to 64x64),")
print("  which is a much simpler baseline than a fine-tuned convolutional network.\n")
print("Refer to your ResNet50 training/evaluation results for comparison of accuracy and other metrics.")
