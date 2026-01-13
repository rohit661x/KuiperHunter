import numpy as np
import pandas as pd
import torch
import copy
from tqdm import tqdm

from src.injection.generator import InjectionPipeline
from src.evaluation.postprocess import extract_candidates
from src.evaluation.tracker import SimpleLinearTracker
from src.evaluation.metrics import compute_metrics, boxes_to_discrete_tracks

class RobustnessSweeper:
    def __init__(self, base_config, model, device):
        """
        Args:
            base_config (dict): Default simulation config.
            model (nn.Module): Trained model.
            device (torch.device): Compute device.
        """
        self.base_config = copy.deepcopy(base_config)
        self.model = model
        self.device = device
        self.tracker = SimpleLinearTracker(
            max_velocity=base_config.get('velocity_range', [1.5,1.5])[1] * 2, # Conservative max
            min_track_len=5
        )

    def run_sweep(self, param_name, param_values, num_samples_per_point=5):
        """
        Run a sweep over a single parameter.
        
        Args:
            param_name (str): Config key to vary (e.g., 'magnitude_range').
            param_values (list): List of values to test.
            num_samples_per_point (int): Samples per value.
            
        Returns:
            pd.DataFrame: Results.
        """
        results = []
        
        for val in tqdm(param_values, desc=f"Sweeping {param_name}"):
            current_config = copy.deepcopy(self.base_config)
            
            # Handle special cases for config setting
            if param_name == 'magnitude':
                current_config['magnitude_range'] = [val, val]
            elif param_name == 'psf':
                current_config['psf_sigma_range'] = [val, val]
            elif param_name == 'snr':
                 # Rough approximation: adjust read noise or flux
                 pass 
            else:
                # Direct set (assuming it's a top level key or handled)
                pass
                
            # Run simulation loop
            pipeline = InjectionPipeline(current_config)
            
            tps, fps, fns = 0, 0, 0
            
            for _ in range(num_samples_per_point):
                # 1. Generate
                frames, _, meta = pipeline.generate_sequence()
                
                # 2. Inference
                inp = torch.from_numpy(np.log1p(frames)).float().unsqueeze(0).unsqueeze(0).to(self.device)
                # Normalize approx
                inp = (inp - inp.mean()) / (inp.std() + 1e-6)
                
                with torch.no_grad():
                    logits = self.model(inp)
                    probs = torch.sigmoid(logits).cpu().numpy()[0, 0] # (T, H, W)
                    
                # 3. Post-proc & Track
                candidates = extract_candidates(probs, threshold=0.5)
                pred_tracks = self.tracker.link_detections(candidates)
                
                # 4. Metrics
                gt_tracks = boxes_to_discrete_tracks(meta['movers'], current_config['sequence_length'])
                metrics = compute_metrics(pred_tracks, gt_tracks)
                
                tps += metrics['tp']
                fps += metrics['fp']
                fns += metrics['fn']
                
            # Aggregate stats for this operating point
            total = tps + fns
            recall = tps / total if total > 0 else 0
            precision = tps / (tps + fps) if (tps + fps) > 0 else 0
            
            results.append({
                param_name: val,
                'recall': recall,
                'precision': precision,
                'f1': 2 * (precision * recall) / (precision + recall + 1e-6),
                'total_tp': tps,
                'total_fp': fps
            })
            
        return pd.DataFrame(results)
