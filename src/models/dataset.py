import torch
from torch.utils.data import Dataset
import numpy as np
from src.injection.generator import InjectionPipeline

class SimulationDataset(Dataset):
    def __init__(self, config, epoch_size=1000):
        """
        Args:
            config (dict): Configuration for InjectionPipeline.
            epoch_size (int): Virtual size of the dataset (number of samples per epoch).
        """
        self.pipeline = InjectionPipeline(config)
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # Generate on the fly
        frames, masks, _ = self.pipeline.generate_sequence()
        
        # Preprocessing
        # 1. Log normalization: log(x + 1) to handle dynamic range
        frames = np.log1p(frames)
        
        # 2. Standardization (rough estimate based on log scale)
        # For simplicity, we can just MinMax scale per image or use global stats
        # Let's use simple global approx for now to correct for background
        frames = (frames - frames.mean()) / (frames.std() + 1e-6)
        
        # 3. To Tensor
        # Input shape: (C, D, H, W) where D is depth (Time)
        # PyTorch 3D layers expect (N, C, D, H, W)
        image_tensor = torch.from_numpy(frames).float().unsqueeze(0) # (1, T, H, W)
        mask_tensor = torch.from_numpy(masks).float().unsqueeze(0)   # (1, T, H, W)
        
        return image_tensor, mask_tensor
