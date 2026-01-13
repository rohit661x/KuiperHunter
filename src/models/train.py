import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.dataset import SimulationDataset
from src.models.unet3d import UNet3D
from src.models.loss import FocalLoss

import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default_simulation.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & Dataloader
    dataset = SimulationDataset(config, epoch_size=config['num_train_sequences'])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0) # num_workers=0 for simplicity/safety

    # Model
    model = UNet3D(n_channels=1, n_classes=1).to(device)

    # Loss & Optimizer
    criterion = FocalLoss(alpha=0.9, gamma=2.0) # High alpha because foreground is very rare
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.6f}")
        
        # Save Checkpoint
        os.makedirs("data/checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"data/checkpoints/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
