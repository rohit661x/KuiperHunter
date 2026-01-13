import torch
import pytest
import shutil
from src.models.unet3d import UNet3D
from src.models.dataset import SimulationDataset

def test_unet3d_shapes():
    model = UNet3D(n_channels=1, n_classes=1)
    # Input: (B, C, T, H, W)
    dummy_input = torch.randn(2, 1, 8, 64, 64)
    output = model(dummy_input)
    # Output: (B, 1, T, H, W)
    assert output.shape == (2, 1, 8, 64, 64)

def test_dataset_output():
    config = {
        'image_size': [32, 32],
        'sequence_length': 6,
        'num_objects_per_sequence': 1,
        'magnitude_range': [10, 10], 
        'flux_zeropoint': 1000.0,
        'velocity_range': [0.0, 1.0],
        'angle_range': [0,0],
        'psf_sigma_range': [1.0, 1.0],
        'noise_level': {'read_noise': 0, 'poisson_noise': False},
        'artifacts': {}
    }
    dataset = SimulationDataset(config, epoch_size=2)
    img, mask = dataset[0]
    
    # Expected: (1, T, H, W)
    assert img.shape == (1, 6, 32, 32)
    assert mask.shape == (1, 6, 32, 32)
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

def test_model_overfit_one_batch():
    """Sanity check: can we reduce loss on a single batch?"""
    device = torch.device('cpu')
    model = UNet3D(n_channels=1, n_classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Random easy input
    inputs = torch.randn(1, 1, 8, 32, 32).to(device)
    targets = (torch.randn(1, 1, 8, 32, 32) > 0).float().to(device)
    
    initial_loss = criterion(model(inputs), targets).item()
    
    for _ in range(10):
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        
    final_loss = criterion(model(inputs), targets).item()
    assert final_loss < initial_loss
