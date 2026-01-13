import numpy as np
from collections import defaultdict

class SimpleLinearTracker:
    def __init__(self, max_velocity=3.0, min_track_len=5, missed_frames=1):
        """
        Args:
            max_velocity (float): Max pixels per frame motion.
            min_track_len (int): Minimum detections to form a valid track.
            missed_frames (int): Allowed missed frames (gaps) in a track.
        """
        self.max_v = max_velocity
        self.min_len = min_len = min_track_len
        self.max_gap = missed_frames
        
    def link_detections(self, candidates):
        """
        Link frame-wise candidates into tracklets.
        
        Args:
            candidates (list of dict): Output from extract_candidates.
            
        Returns:
            list of list of dict: List of tracks (each track is a list of detection dicts).
        """
        # Sort by time
        candidates = sorted(candidates, key=lambda c: c['t'])
        
        # Group by time
        framewise = defaultdict(list)
        for c in candidates:
            framewise[c['t']].append(c)
            
        active_tracks = [] # List of lists
        final_tracks = []
        
        # Iterate through time
        if not candidates:
            return []
            
        max_t = candidates[-1]['t']
        
        for t in range(max_t + 1):
            current_dets = framewise[t]
            
            # Try to extend active tracks
            # Simple greedy assignment
            # Calculate costs matrix or just greedy nearest neighbor
            
            unassigned_dets = current_dets[:]
            
            # Sort active tracks by length (prefer extending long tracks)
            active_tracks.sort(key=len, reverse=True)
            
            for track in active_tracks:
                if track[-1]['t'] < t - self.max_gap - 1:
                    # Track is dead (too many missed frames)
                    continue
                
                # Predict next position (Linear Constant Velocity)
                last = track[-1]
                if len(track) >= 2:
                    prev = track[-2]
                    dt = last['t'] - prev['t']
                    vx = (last['x'] - prev['x']) / dt
                    vy = (last['y'] - prev['y']) / dt
                else:
                    vx, vy = 0, 0 # Assume static if only 1 point? Or ambiguous.
                
                dt_pred = t - last['t']
                pred_x = last['x'] + vx * dt_pred
                pred_y = last['y'] + vy * dt_pred
                
                # Search constraint
                # If we have velocity, search radius is small error margin
                # If no velocity (first point), search radius is max_v * dt
                
                radius = self.max_v * dt_pred if len(track) < 2 else (self.max_v * 0.5 + 1.0) # tighter if established
                
                # Find best match
                best_match = None
                best_dist = float('inf')
                
                for det in unassigned_dets:
                    dist = np.sqrt((det['x'] - pred_x)**2 + (det['y'] - pred_y)**2)
                    if dist <= radius and dist < best_dist:
                        best_dist = dist
                        best_match = det
                
                if best_match:
                    track.append(best_match)
                    unassigned_dets.remove(best_match)
            
            # Init new active tracks from unassigned
            for det in unassigned_dets:
                active_tracks.append([det])
                
        # Filter final tracks
        valid_tracks = [t for t in active_tracks if len(t) >= self.min_len]
        return valid_tracks
