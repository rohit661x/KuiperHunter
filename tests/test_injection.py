import numpy as np
import pytest
from src.injection.generator import InjectionPipeline

@pytest.fixture
def test_config():
    return {
        'image_size': [64, 64],
        'sequence_length': 5,
        'num_objects_per_sequence': 1,
        'magnitude_range': [0, 0], # High flux for visibility
        'flux_zeropoint': 1000.0,
        'velocity_range': [1.0, 1.0],
        'angle_range': [0, 0], # Move strictly right
        'psf_sigma_range': [1.0, 1.0],
        'noise_level': {
            'read_noise': 0.0, 
            'poisson_noise': False
        },
        'artifacts': {
            'hot_pixels_prob': 0.0
        }
    }

def test_pipeline_output_shapes(test_config):
    pipeline = InjectionPipeline(test_config)
    frames, masks, meta = pipeline.generate_sequence()
    
    T, H, W = test_config['sequence_length'], *test_config['image_size']
    assert frames.shape == (T, H, W)
    assert masks.shape == (T, H, W)
    assert len(meta['movers']) == 1

def test_trajectory_logic(test_config):
    # Force specific trajectory params
    # We can't easily mock the random inside the class without refactor, 
    # but we can check if the object moves.
    
    # Configure so it's bright and moves right
    test_config['velocity_range'] = [2.0, 2.0] # 2 pix/frame
    test_config['angle_range'] = [0, 0] # +x direction
    
    pipeline = InjectionPipeline(test_config)
    frames, _, meta = pipeline.generate_sequence()
    
    mover = meta['movers'][0]
    x0, y0 = mover['x0'], mover['y0']
    vx, vy = mover['vx'], mover['vy']
    
    assert np.isclose(vx, 2.0)
    assert np.isclose(vy, 0.0)
    
    # Check if we can find flux peaks near expected positions
    for t in range(5):
        xt_expected = x0 + vx * t
        yt_expected = y0 + vy * t
        
        # Only check if expected is well within bounds
        if 5 < xt_expected < 59 and 5 < yt_expected < 59:
            # Look at a small window around expected location
            iy, ix = int(yt_expected), int(xt_expected)
            patch = frames[t, iy-2:iy+3, ix-2:ix+3]
            # Max of patch should be significant
            assert patch.max() > 100 # Star should be bright
