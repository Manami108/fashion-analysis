import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the same validation transform used during training.
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

def extract_segmented_region(image_tensor, mask_tensor):
    """Multiply the image tensor by the mask tensor."""
    return image_tensor * mask_tensor

def get_dummy_mask(image_size=(1, 256, 256)):
    """Create a dummy segmentation mask (all ones)."""
    return torch.ones(image_size)

# (A) DenseNet-based Model for Clothing (Single-label Classification)
num_cloth_classes = 13
def get_cloth_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    # Replace the classifier with a custom head.
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_cloth_classes)
    )
    return model

# (B) ResNet50-based Model for Fashion Style (Single-label Classification)
num_style_classes = 14  
def get_style_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Replace the final fully connected layer with a linear layer.
    model.fc = nn.Linear(model.fc.in_features, num_style_classes)
    return model


cloth_model_path = ""
style_model_path = ""

# Load the DenseNet clothing model.
cloth_model = get_cloth_model().to(device)
cloth_model.load_state_dict(torch.load(cloth_model_path, map_location=device))
cloth_model.eval()

# Load the ResNet50 fashion style model.
style_model = get_style_model().to(device)
style_model.load_state_dict(torch.load(style_model_path, map_location=device))
style_model.eval()

# Define the labels
cloth_labels = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "shorts", "trousers", "skirt", "short sleeve dress",
    "long sleeve dress", "vest dress", "sling dress"
]

style_labels = [
    "conservative", "dressy", "ethnic", "fairy", "feminine", "gal", "girlish", 
    "kireime-casual", "lolita", "mode", "natural", "retro", "rock", "street"
]

def predict_image(image_path):

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None, None
    input_tensor = transform_val(image)  

    mask_tensor = get_dummy_mask(image_size=(1, 256, 256))

    segmented_image = extract_segmented_region(input_tensor, mask_tensor)
    
    input_batch = segmented_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # For clothing: argmax over logits are used.
        cloth_logits = cloth_model(input_batch)  
        cloth_pred_idx = torch.argmax(cloth_logits, dim=1).item()
        predicted_cloth = cloth_labels[cloth_pred_idx] if cloth_pred_idx < len(cloth_labels) else "Unknown"
        
        # For fashion style: similarly, argmax over logits are used.
        style_logits = style_model(input_batch) 
        style_pred_idx = torch.argmax(style_logits, dim=1).item()
        predicted_style = style_labels[style_pred_idx] if style_pred_idx < len(style_labels) else "Unknown"
    
    return predicted_cloth, predicted_style

# Test with unseen data
if __name__ == "__main__":
    unseen_folder = ""

    image_files = [f for f in os.listdir(unseen_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    results = {}
    for file_name in image_files:
        image_path = os.path.join(unseen_folder, file_name)
        predicted_cloth, predicted_style = predict_image(image_path)
        if predicted_cloth is not None and predicted_style is not None:
            results[file_name] = {"cloth": predicted_cloth, "style": predicted_style}
            print(f"Image: {file_name}")
            print(f"  Predicted Clothing Type: {predicted_cloth}")
            print(f"  Predicted Fashion Style: {predicted_style}\n")
    
 
