import os

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import umap.umap_ as umap

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoImageProcessor

import clip
from diffusers import StableDiffusionInstructPix2PixPipeline

from attacker_network import AttackerNetwork
from preprocess import preprocess

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# --Create dataloader--
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root='dogs_data/Images/', transform=transform)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# -- Import models --

# Load Dinov2 model
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-small")
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dinov2_model.to(device)

# Load CLIP model
clip_model, _ = clip.load("ViT-L/14", device=device)

# Load InstructPix2Pix model
instruct_pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
            ).to(device)

instruct_pix2pix.unet.enable_gradient_checkpointing()

# Attacker network
attacker = AttackerNetwork(
    clip_model=clip_model,
    pix2pix_model=instruct_pix2pix,
    dinov2_model=dinov2_model,
    device=device
)


# -- Preprocess data --
subset, centroids, starting_class, target_class = preprocess(dataloader,
                                                              dinov2_model)

target_centroid = torch.tensor(centroids[target_class]).to(device)
target_centroid = target_centroid.unsqueeze(0)
target_centroid = F.normalize(target_centroid, dim=1)


# --Training loop--
n_epochs = 1
distance_metric = "cosine"

optimizer = torch.optim.Adam(attacker.parameters(), lr=1e-3)
attacker.train()

for epoch in range(n_epochs):
    for image, _ in tqdm(subset, desc=f"Epoch {epoch+1}/{n_epochs}"):
        image = image.to(device)

        output = attacker(image)[0]

        if distance_metric == "euclidean":
            dist = torch.norm(output - target_centroid, dim=1)

            loss = - torch.mean(dist)
        
        elif distance_metric == "cosine":
            output_norm = F.normalize(output, dim=1)
            dist = torch.nn.functional.cosine_similarity(output_norm, target_centroid)

            loss = - torch.mean(dist)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item():.4f}")





