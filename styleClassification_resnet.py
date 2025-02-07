import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix, classification_report


base_dir = ""

class_names = [
    "conservative", "dressy", "ethnic", "fairy", "feminine", "gal", "girlish", 
    "kireime-casual", "lolita", "mode", "natural", "retro", "rock", "street"
]

print("Number of images per class:")
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
for cls in class_names:
    class_path = os.path.join(base_dir, cls)
    count = sum([1 for file in os.listdir(class_path) if file.lower().endswith(image_extensions)])
    print(f"  {cls:15s}: {count}")


IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# ImageNet normalization parameters is defined 
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# Define transforms for training and validation
train_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Define the validation check function to filter out non-image files and undesired filenames.
def is_valid_file(path):
    if os.path.basename(path).startswith("._"):
        return False
    return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

full_dataset = datasets.ImageFolder(
    root=base_dir, 
    transform=train_transforms,
    is_valid_file=is_valid_file
)

print("\nClass to index mapping:")
print(full_dataset.class_to_idx)

# Split dataset into training (80%) and validation (20%)
dataset_size = len(full_dataset)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size

torch.manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

num_classes = len(full_dataset.classes)
print("\nNumber of classes:", num_classes)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
# Trains the model using the specified optimizer, scheduler, and dataloaders, with early stopping based on validation loss.
# Sets up a pretrained ResNet50 model with a custom fully connected (FC) layer.
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

# Freezes all layers except the final fully connected (FC) layer during initial training.
for name, param in model_ft.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7)


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=100, patience=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = dataloaders['train']
            else:
                model.eval()
                dataloader = dataloaders['val']

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print("Validation loss decreased, saving model...")
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val loss: {best_loss:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history

dataloaders = {'train': train_loader, 'val': val_loader}

print("\nStarting initial training (frozen backbone)...")
model_ft, history = train_model(model_ft, criterion, optimizer_ft, scheduler, dataloaders, device, num_epochs=20, patience=5)

# Unfreezes the last ResNet block (layer4) and the FC layer for fine-tuning.
for name, param in model_ft.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-5)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7)

print("\nStarting fine-tuning (unfrozen last layers)...")
model_ft, ft_history = train_model(model_ft, criterion, optimizer_ft, scheduler, dataloaders, device, num_epochs=10, patience=5)

combined_history = {}
for key in history.keys():
    combined_history[key] = history[key] + ft_history.get(key, [])

model_ft.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
classes_idx_to_name = {v: k for k, v in full_dataset.class_to_idx.items()}

def plot_confusion_matrix(cm, classes, normalize=True, title="Confusion Matrix (%)", cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True) 
        cm = np.nan_to_num(cm) * 100  
        print("Normalized confusion matrix (in %)")
    else:
        print("Confusion matrix, without normalization")
    print(cm)
    
    plt.figure(figsize=(12, 10))

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    fmt = 'd' if normalize else 'd'  
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(int(round(cm[i, j])), fmt), 
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



class_labels = [classes_idx_to_name[i] for i in range(num_classes)]
plot_confusion_matrix(cm, classes=class_labels, title="Confusion Matrix (%)")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_labels))


def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(combined_history)

# Saves the best model (based on validation loss) to a file.
model_save_path = ""
torch.save(model_ft.state_dict(), model_save_path)
print(f"\nBest model saved to '{model_save_path}'")

