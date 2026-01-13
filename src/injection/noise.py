import numpy as np

def add_gaussian_noise(image, sigma):
    """
    Add Gaussian read noise to the image.
    
    Args:
        image (np.ndarray): Input image.
        sigma (float): Standard deviation of noise.
        
    Returns:
        np.ndarray: Noisy image.
    """
    noise = np.random.normal(0, sigma, image.shape)
    return image + noise

def add_poisson_noise(image):
    """
    Add Poisson shot noise.
    Assumes image values are in photon counts (electrons).
    
    Args:
        image (np.ndarray): Input image.
        
    Returns:
        np.ndarray: Noisy image.
    """
    # Poisson noise requires non-negative values
    img_safe = np.maximum(image, 0)
    noisy = np.random.poisson(img_safe).astype(float)
    return noisy

def add_hot_pixels(image, prob, saturation=1e5):
    """Randomly set pixels to saturation value."""
    mask = np.random.random(image.shape) < prob
    image[mask] = saturation
    return image
