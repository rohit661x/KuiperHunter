import numpy as np

def compute_metrics(pred_tracks, gt_tracks, distance_threshold=3.0):
    """
    Compute Precision, Recall for TRACKS.
    A track is considered matched if it roughly overlaps with a GT track.
    Simple criteria: Centroid distance of the start/end points or IoU of frames.
    
    For simplicity here: A pred track is a True Positive if it matches a GT track
    in >50% of its frames within distance_threshold.
    
    Args:
        pred_tracks (list of list of dict): Predicted tracks.
        gt_tracks (list of list of dict): Ground truth tracks (from metadata).
    
    Returns:
        dict: {'precision': float, 'recall': float, 'fp': int, 'tp': int, 'fn': int}
    """
    
    # Convert GT format if needed. 
    # Usually GT is simplified to parameters, but we might have discrete points or eq.
    # Let's assume gt_tracks in similar format or we evaluate "is this pred track valid?"
    
    # If gt_tracks comes from 'movers' metadata, we need to generate discrete points to compare
    # Or we construct a "is_valid(track)" function based on discrete points.
    
    tp = 0
    fp = 0
    
    # Track matching (Greedy)
    # 1. For each Pred Track, check if it matches ANY GT track
    used_gt_indices = set()
    
    for pred in pred_tracks:
        # A track is a set of (t, x, y)
        is_match = False
        matched_gt_idx = -1
        
        # Check against all GT
        for idx, gt in enumerate(gt_tracks):
            # gt is list of dicts or object with frame data
            # Let's count matching frames
            matches = 0
            for p_point in pred:
                t = p_point['t']
                # find gt point at time t
                # Assuming gt is a list of points too
                gt_point = next((g for g in gt if g['t'] == t), None)
                if gt_point:
                    dist = np.sqrt((p_point['x'] - gt_point['x'])**2 + (p_point['y'] - gt_point['y'])**2)
                    if dist < distance_threshold:
                        matches += 1
            
            # Criteria: Match > 50% of the PREDICTED length (clean output) 
            # AND > 30% of GT length (coverage)
            if matches / len(pred) > 0.5 and matches / len(gt) > 0.3:
                is_match = True
                matched_gt_idx = idx
                break
        
        if is_match:
            tp += 1
            used_gt_indices.add(matched_gt_idx)
        else:
            fp += 1
            
    fn = len(gt_tracks) - len(used_gt_indices)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'tp': tp, 
        'fp': fp, 
        'fn': fn
    }

def boxes_to_discrete_tracks(metadata, T):
    """
    Helper to convert generator 'movers' metadata into discrete x,y points for evaluation.
    """
    tracks = []
    for m in metadata: # movers list
        track = []
        x0, y0, vx, vy = m['x0'], m['y0'], m['vx'], m['vy']
        for t in range(T):
            xt = x0 + vx * t
            yt = y0 + vy * t
            track.append({'t': t, 'x': xt, 'y': yt})
        tracks.append(track)
    return tracks
