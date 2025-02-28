import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from train_utils import train_utils
from transformers import CLIPProcessor, CLIPModel

# ✅ Define activations as a global dictionary
activations = {}
device = torch.device('cuda')
# Standard ADAM optimizer
# Note that EMA parametesr are not optimized

#Encode an image:

#Encode text descriptions

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = model
inputs = processor(text=['cable'], return_tensors="pt", padding="max_length")

with torch.no_grad():
    outputs = model.text_model(
                            return_dict=True,   
                            output_hidden_states=True,
                            output_attentions=False,
                            **inputs,
                        )
embedding1 = outputs.pooler_output[0]
# Define a hook function to store feature maps
def get_activation(name):
    def hook(model, input, output):
        print(f"✅ Hook triggered for {name}!")
        activations[name] = output.detach()  # ✅ Store the output feature map
    return hook

# Load regular ResNet
resnet = train_utils().get_resnet("resnet34")
resnet.eval()

# Load FiLM-conditioned ResNet (Assuming you have a `get_resnet` function)
film_resnet = train_utils().get_resnet_FILM("resnet34")  # Your FiLM-conditioned model
film_resnet.eval()

# ✅ Register hooks to extract feature maps (Choose `layer3` or `layer4`)
resnet.layer4.register_forward_hook(get_activation("resnet"))
film_resnet.layer4[1].register_forward_hook(get_activation("film_resnet"))

# Load a sample image and preprocess it
from torchvision import transforms
from PIL import Image

image_path = "/home/lm-2023/jeon_team_ws/lbr-stack/src/data_collection/data_collection/NIST_new/t65/images_B/camera_B_59.png"  # Use any test image
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Generate a dummy language embedding (CLIP or other embeddings)
lang_emb = torch.randn(1, 512)  # Example CLIP embedding

# ✅ Pass image through both models to populate `activations`
with torch.no_grad():
    resnet(input_tensor)  # Regular ResNet
    film_resnet(input_tensor, embedding1)  # FiLM-Conditioned ResNet

# ✅ Extract feature maps
resnet_feature_map = activations["resnet"].squeeze(0)  # (C, H, W)
film_feature_map = activations["film_resnet"].squeeze(0)  # (C, H, W)

# ✅ Check if activations exist before visualizing
assert "resnet" in activations, "ResNet activations not found!"
assert "film_resnet" in activations, "FiLM ResNet activations not found!"

# Select a feature channel to visualize
channel_idx = 10  # Change this to visualize different features
# Normalize each feature map individually
def normalize_feature_map(feature_map):
    feature_map -= feature_map.min()  # Shift min to 0
    if feature_map.max() > 0:  # Avoid division by zero
        feature_map /= feature_map.max()  # Scale max to 1
    return feature_map

resnet_activation = normalize_feature_map(resnet_feature_map[channel_idx].cpu().numpy())

film_activation = normalize_feature_map(film_feature_map[channel_idx].cpu().numpy())

# Normalize for better visualization
resnet_activation -= resnet_activation.min()
resnet_activation /= resnet_activation.max()
film_activation -= film_activation.min()
film_activation /= film_activation.max()

import cv2

# Resize activation to match the input image size
resnet_activation_resized = cv2.resize(resnet_activation, (320, 240))
film_resnet_activation_resized = cv2.resize(film_activation, (320, 240))

# Convert to heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(film_resnet_activation_resized, 0, 1)), cv2.COLORMAP_JET)
# Convert image to numpy (RGB format)
image_np = np.array(image.resize((320, 240))).astype(np.uint8)


# Overlay heatmap on the original image
overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

plt.imshow(overlay)
plt.title("Feature Map Overlay")
plt.axis("off")
plt.show()
# film_activation -= film_activation.min()
# film_activation /= film_activation.max()
