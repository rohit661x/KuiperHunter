import numpy as np
from scipy.stats import multivariate_normal

def gaussian_2d_kernel(shape, sigma, center=None):
    """
    Generate a 2D Gaussian kernel.
    
    Args:
        shape (tuple): (height, width) of the kernel.
        sigma (float): Standard deviation of the Gaussian.
        center (tuple, optional): (x, y) center of the Gaussian. Defaults to center of shape.
    
    Returns:
        np.ndarray: Normalized 2D Gaussian kernel.
    """
    m, n = shape
    if center is None:
        y0, x0 = m // 2, n // 2
    else:
        x0, y0 = center
    
    y, x = np.mgrid[0:m, 0:n]
    pos = np.dstack((x, y))
    mean = [x0, y0]
    cov = [[sigma**2, 0], [0, sigma**2]]
    
    rv = multivariate_normal(mean, cov)
    kernel = rv.pdf(pos)
    
    # Normalize
    return kernel / kernel.sum()

def render_point_source(image, x, y, flux, sigma, kernel_size=21):
    """
    Render a point source onto an image with a Gaussian PSF.
    
    Args:
        image (np.ndarray): Target image to add source to (H, W).
        x (float): x-coordinate of source center.
        y (float): y-coordinate of source center.
        flux (float): Total flux of the source.
        sigma (float): PSF width (sigma).
        kernel_size (int): Size of the PSF stamp to render.
        
    Returns:
        np.ndarray: Image with added source.
    """
    H, W = image.shape
    r = kernel_size // 2
    
    # Integer bounds for the patch
    ix, iy = int(round(x)), int(round(y))
    
    # Check if completely out of bounds
    if ix < -r or ix >= W + r or iy < -r or iy >= H + r:
        return image

    # Generate kernel centered at subpixel offset relative to the patch center
    # The patch is centered at (ix, iy)
    # The source is at (x, y) = (ix + dx, iy + dy)
    dx = x - ix
    dy = y - iy
    
    # Evaluate Gaussian on the patch grid
    # Grid coordinates relative to (ix, iy)
    y_grid, x_grid = np.mgrid[-r:r+1, -r:r+1]
    
    # Gaussian function: exp( -((x-dx)^2 + (y-dy)^2) / (2*sigma^2) )
    kernel = np.exp( -((x_grid - dx)**2 + (y_grid - dy)**2) / (2 * sigma**2) )
    kernel = kernel / (2 * np.pi * sigma**2) # Normalization
    
    # Scale by flux
    stamp = kernel * flux
    
    # Paste into image with bounds checking
    # Source range (in stamp coordinates)
    sx_start, sx_end = 0, kernel_size
    sy_start, sy_end = 0, kernel_size
    
    # Target range (in image coordinates)
    tx_start, tx_end = ix - r, ix + r + 1
    ty_start, ty_end = iy - r, iy + r + 1
    
    # Clip to image boundaries
    if tx_start < 0:
        sx_start += -tx_start
        tx_start = 0
    if ty_start < 0:
        sy_start += -ty_start
        ty_start = 0
    if tx_end > W:
        sx_end -= (tx_end - W)
        tx_end = W
    if ty_end > H:
        sy_end -= (ty_end - H)
        ty_end = H
        
    if sx_start < sx_end and sy_start < sy_end:
        image[ty_start:ty_end, tx_start:tx_end] += stamp[sy_start:sy_end, sx_start:sx_end]
        
    return image
