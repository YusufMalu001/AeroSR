import os
import cv2
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from sr_model import ESPCN
from degrade import degrade_image

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, 'data')
ORIGINAL_DIR = os.path.join(DATA_DIR, 'original')
WEIGHTS_DIR = os.path.join(WORKSPACE_DIR, 'weights')
TRAIN_LIST = os.path.join(DATA_DIR, 'train_images.txt')

class DIORPatchDataset(Dataset):
    def __init__(self, image_list_file, patches_per_image=50, patch_size=64, scale_factor=4):
        self.image_paths = []
        if os.path.exists(image_list_file):
            with open(image_list_file, 'r') as f:
                self.image_paths = [os.path.join(ORIGINAL_DIR, line.strip()) for line in f if line.strip()]
                
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        
        # Load images into memory to speed up training (200 images * 800x800x3 is ~384MB)
        print("Loading training images into memory...")
        self.images = []
        for path in tqdm(self.image_paths):
            img = cv2.imread(path)
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images.append(img)
                
        print(f"Loaded {len(self.images)} images.")

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        img = self.images[img_idx]
        
        h, w = img.shape[:2]
        
        # Random crop
        y = np.random.randint(0, h - self.patch_size)
        x = np.random.randint(0, w - self.patch_size)
        
        hr_patch = img[y:y+self.patch_size, x:x+self.patch_size]
        
        # Degrade to LR patch
        # degrade_image returns (LR, Bicubic). We only want LR.
        lr_patch, _ = degrade_image(hr_patch, scale_factor=self.scale_factor, noise_sigma=15, blur_kernel=5, apply_haze=True)
        
        # Normalize and convert to tensor (C, H, W)
        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).float() / 255.0
        lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).float() / 255.0
        
        return lr_tensor, hr_tensor

def train_model(epochs=15, batch_size=32):
    if not os.path.exists(TRAIN_LIST):
        print(f"Training list {TRAIN_LIST} not found. Ensure prepare_dataset.py ran successfully.")
        return

    dataset = DIORPatchDataset(TRAIN_LIST, patches_per_image=50, patch_size=64)
    if len(dataset) == 0:
        print("No training data found.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    device = torch.device('cpu') # Target is CPU-only
    print(f"Using device: {device}")
    
    model = ESPCN(upscale_factor=4).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    loss_history = []
    
    print(f"Starting training on {len(dataset)} patches for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for lr, hr in pbar:
            lr = lr.to(device)
            hr = hr.to(device)
            
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append({'epoch': epoch, 'loss': avg_loss})
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
    # Save model
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'espcn_dior.pt'))
    
    # Save loss history
    with open(os.path.join(DATA_DIR, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f)
        
    print("Training complete. Weights and history saved.")

if __name__ == '__main__':
    train_model(epochs=15)
