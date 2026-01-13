import numpy as np
from scipy.ndimage import label, center_of_mass, maximum_position

def extract_candidates(prob_map, threshold=0.5):
    """
    Extract candidate detections from a probability map.
    
    Args:
        prob_map (np.ndarray): (T, H, W) probability map (0-1).
        threshold (float): Threshold for detection.
        
    Returns:
        list of dict: List of candidates, each dict containing:
            - t: frame index
            - x, y: centroid coordinates
            - peak: peak probability value
    """
    T, H, W = prob_map.shape
    candidates = []
    
    # Process each frame
    for t in range(T):
        frame_probs = prob_map[t]
        mask = frame_probs > threshold
        
        if not np.any(mask):
            continue
            
        # Label connected components
        labeled_array, num_features = label(mask)
        
        if num_features == 0:
            continue
            
        # Get centroids
        centroids = center_of_mass(frame_probs, labeled_array, range(1, num_features+1))
        
        # Get peak values (rough approx: max value in the component)
        # For simplicity, we can just look up value at integer centroid or use maximum_position
        # Let's iterate components
        for i in range(1, num_features+1):
            component_mask = (labeled_array == i)
            peak_val = np.max(frame_probs[component_mask])
            
            # center_of_mass returns (y, x)
            cy, cx = centroids[i-1]
            
            candidates.append({
                't': t,
                'x': cx,
                'y': cy,
                'score': float(peak_val)
            })
            
    return candidates
