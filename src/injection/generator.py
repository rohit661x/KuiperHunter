import numpy as np
import yaml
from src.injection.psf import render_point_source
from src.injection.background import SyntheticStarfield, BackgroundLoader
from src.injection.noise import add_gaussian_noise, add_poisson_noise, add_hot_pixels

class InjectionPipeline:
    def __init__(self, config):
        """
        Initialize the pipeline with configuration dictionary.
        Args:
            config (dict): Configuration parameters.
        """
        self.config = config
        self.H, self.W = config['image_size']
        self.T = config['sequence_length']
        
        # Background Generator Selection
        if self.config.get('use_real_backgrounds', False):
            self.bg_generator = BackgroundLoader()
        else:
            self.bg_generator = SyntheticStarfield(
                shape=(self.H, self.W),
                flux_range=(100, 5000) # Default for background stars
            )

    def generate_trajectory(self):
        """Sample random linear trajectory parameters."""
        # Random initial position
        x0 = np.random.uniform(0, self.W)
        y0 = np.random.uniform(0, self.H)
        
        # Random velocity
        v_min, v_max = self.config['velocity_range']
        speed = np.random.uniform(v_min, v_max)
        angle = np.random.uniform(*self.config['angle_range'])
        theta = np.deg2rad(angle)
        
        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)
        
        return x0, y0, vx, vy

    def generate_sequence(self):
        """
        Generate a single sequence with injected objects.
        Returns:
            frames (np.ndarray): (T, H, W)
            masks (np.ndarray): (T, H, W) ground truth binary mask
            metadata (dict): injection parameters
        """
        # 1. Generate Static Background
        # Pass shape explicitly for Real loader cropping
        frames = self.bg_generator.get_sequence(self.T, shape=(self.H, self.W)) # (T, H, W)
        masks = np.zeros((self.T, self.H, self.W), dtype=float)
        
        # 2. Generate Movers
        num_movers = self.config['num_objects_per_sequence']
        movers = []
        
        for _ in range(num_movers):
            x0, y0, vx, vy = self.generate_trajectory()
            
            # Flux
            mag_min, mag_max = self.config['magnitude_range']
            mag = np.random.uniform(mag_min, mag_max)
            # Flux conversion: F = F0 * 10^(-0.4 * m)
            f0 = self.config.get('flux_zeropoint', 1000.0)
            flux = f0 * (10 ** (-0.4 * mag))
            
            movers.append({
                'id': len(movers),
                'x0': x0, 'y0': y0,
                'vx': vx, 'vy': vy,
                'flux': flux,
                'mag': mag
            })
            
            # 3. Inject into frames
            for t in range(self.T):
                xt = x0 + vx * t
                yt = y0 + vy * t
                
                # Check bounds loosely (don't precise clip here, render handles it)
                # PSF Variation
                sigma_min, sigma_max = self.config['psf_sigma_range']
                sigma_t = np.random.uniform(sigma_min, sigma_max)
                
                # Render Object
                # Note: modifying frames in-place (slice)
                render_point_source(frames[t], xt, yt, flux, sigma_t, kernel_size=15)
                
                # Render Mask (simply a blob or single pixel)
                # For simplified ground truth, we can put a Gaussian blob or boolean disk
                # Here using the same render function but with unit flux for heatmap style
                render_point_source(masks[t], xt, yt, 1.0, sigma_t, kernel_size=15)

        # 4. Add Noise & Artifacts
        noise_cfg = self.config.get('noise_level', {})
        
        # Poisson
        if noise_cfg.get('poisson_noise', True):
            # frames should be non-negative for poisson
            frames = add_poisson_noise(frames)
            
        # Read Noise (Gaussian)
        read_noise = noise_cfg.get('read_noise', 0.0)
        if read_noise > 0:
            frames = add_gaussian_noise(frames, read_noise)
            
        # Hot Pixels
        artifacts_cfg = self.config.get('artifacts', {})
        hp_prob = artifacts_cfg.get('hot_pixels_prob', 0.0)
        if hp_prob > 0:
            for t in range(self.T):
                add_hot_pixels(frames[t], hp_prob)

        return frames, masks, {'movers': movers}
