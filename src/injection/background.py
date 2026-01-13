import numpy as np
import os
from astropy.io import fits
from astropy.utils.data import download_file
from src.injection.psf import render_point_source

class BackgroundLoader:
    """Loads real background images from FITS files."""
    def __init__(self, cache_dir='data/real'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # List of candidate URLs for sample FITS data
        self.candidate_urls = [
            'https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits', # HST Eagle Nebula (Reliable)
            'http://data.astropy.org/tutorials/FITS-images/HorseHead.fits',
        ]
        self.cached_path = None

    def _ensure_sample(self):
        if self.cached_path and os.path.exists(self.cached_path):
            return self.cached_path
        
        for url in self.candidate_urls:
            print(f"Attempting to download sample FITS from {url}...")
            try:
                self.cached_path = download_file(url, cache=True, timeout=10)
                print("Download successful.")
                return self.cached_path
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
                
        raise RuntimeError("Could not download any sample FITS files. Check internet connection.")

    def get_crop(self, shape):
        path = self._ensure_sample()
        
        with fits.open(path) as hdul:
            # Find the first HDU with image data
            data = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    data = hdu.data
                    break
            
            if data is None:
                raise ValueError("No image data found in FITS file.")

        # Handle >2 dimensions (e.g. RGB or cubes)
        if data.ndim > 2:
            print(f"Warning: FITS data has shape {data.shape}. Taking the first slice.")
            while data.ndim > 2:
                data = data[0]

        H_real, W_real = data.shape
        h, w = shape
        
        if H_real < h or W_real < w:
            raise ValueError(f"Real image ({H_real}x{W_real}) smaller than requested ({h}x{w})")
            
        # Random crop
        x = np.random.randint(0, W_real - w + 1)
        y = np.random.randint(0, H_real - h + 1)
        
        crop = data[y:y+h, x:x+w].astype(float)
        
        # Normalization for injection compatibility
        # We scale the image so that the noise standard deviation matches a 'canonical' value (e.g. 5.0)
        # This ensures our fixed-magnitude injections (which expect a certain noise floor) are visible.
        
        bg_median = np.median(crop)
        crop -= bg_median # Zero-center
        
        bg_std = np.std(crop)
        if bg_std == 0: bg_std = 1.0
        
        # Target noise sigma = 5.0 (consistent with default generator config)
        scale_factor = 5.0 / bg_std
        crop *= scale_factor
        
        return crop

    def get_sequence(self, length, shape):
        """Returns a sequence of the same static real background (T, H, W)."""
        # Get one crop
        static_bg = self.get_crop(shape)
        
        # Repeat it
        return np.repeat(static_bg[np.newaxis, :, :], length, axis=0)

class SyntheticStarfield:
    """Generates synthetic static starfields (FR-1 Option B)."""
    def __init__(self, density=0.001, flux_range=(100, 10000), shape=(128, 128)):
        """
        Args:
            density (float): Stars per pixel probability.
            flux_range (tuple): (min_flux, max_flux).
            shape (tuple): default frame shape.
        """
        self.density = density
        self.flux_range = flux_range
        self.shape = shape
        self.static_field = None

    def generate_static_field(self, shape=None):
        if shape is None:
            shape = self.shape
        H, W = shape
        
        # Determine number of stars
        num_stars = int(H * W * self.density)
        
        # Generate star positions and fluxes
        xs = np.random.uniform(0, W, num_stars)
        ys = np.random.uniform(0, H, num_stars)
        fluxes = np.random.uniform(self.flux_range[0], self.flux_range[1], num_stars)
        
        # Render static stars
        # For simplicity, use a constant PSF for background stars or randomize slightly
        image = np.zeros(shape)
        for x, y, f in zip(xs, ys, fluxes):
            sigma = np.random.uniform(0.8, 1.5) # Background stars have varying PSF
            render_point_source(image, x, y, f, sigma, kernel_size=11)
            
        self.static_field = image
        return image

    def get_sequence(self, length, shape=None):
        """Returns a sequence of the same static background (T, H, W)."""
        if self.static_field is None or (shape is not None and self.static_field.shape != shape):
            self.generate_static_field(shape)
        
        H, W = self.static_field.shape
        return np.repeat(self.static_field[np.newaxis, :, :], length, axis=0)
