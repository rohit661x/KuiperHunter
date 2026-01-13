import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.injection.generator import InjectionPipeline
from src.models.unet3d import UNet3D
from src.evaluation.postprocess import extract_candidates
from src.evaluation.tracker import SimpleLinearTracker
from src.evaluation.metrics import boxes_to_discrete_tracks, compute_metrics

def main():
    parser = argparse.ArgumentParser(description="KuiperHunter End-to-End Demo")
    parser.add_argument('--config', type=str, default='config/smoke_test.yaml', help='Path to config')
    parser.add_argument('--checkpoint', type=str, default='data/checkpoints/model_epoch_5.pth', help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='demo_result.gif', help='Output GIF path')
    parser.add_argument('--no-gif', action='store_true', help='Skip GIF generation')
    parser.add_argument('--view', action='store_true', help='Open output file after generation')
    args = parser.parse_args()

    # 1. Setup
    print("Initializing...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    mode = "REAL FITS DATA" if config.get('use_real_backgrounds', False) else "SYNTHETIC DATA"
    print(f"--- Running in {mode} mode ---")

    # 2. Components
    # Generators
    pipeline = InjectionPipeline(config)
    
    # Model
    model = UNet3D(n_channels=1, n_classes=1).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded model from {args.checkpoint}")
    else:
        print("Warning: Checkpoint not found, using random weights (expect bad results!)")
    model.eval()
    
    # Tracker
    tracker = SimpleLinearTracker(max_velocity=5.0, min_track_len=min(3, config['sequence_length']))

    # 3. Execution Loop
    print("\n--- Generating Sequence ---")
    frames, _, meta = pipeline.generate_sequence()
    print(f"Generated {len(meta['movers'])} movers.")
    for m in meta['movers']:
        print(f"  Mover {m['id']}: Mag={m['mag']:.2f}, Vel=({m['vx']:.2f}, {m['vy']:.2f})")

    print("\n--- Running Inference ---")
    # Preprocess
    inp = torch.from_numpy(np.log1p(frames)).float().unsqueeze(0).unsqueeze(0).to(device)
    inp = (inp - inp.mean()) / (inp.std() + 1e-6)
    
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0] # (T, H, W)
        
    print("\n--- Tracking ---")
    candidates = extract_candidates(probs, threshold=0.5)
    print(f"Found {len(candidates)} candidate blobs across all frames.")
    
    pred_tracks = tracker.link_detections(candidates)
    print(f"Linked into {len(pred_tracks)} tracks.")
    
    # 4. Evaluation
    gt_tracks = boxes_to_discrete_tracks(meta['movers'], config['sequence_length'])
    metrics = compute_metrics(pred_tracks, gt_tracks, distance_threshold=3.0)
    
    print("\n--- Results ---")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    
    # 5. Visualization (GIF)
    if not args.no_gif:
        print(f"\nSaving visualization to {args.output}...")
        save_gif(frames, probs, pred_tracks, gt_tracks, args.output)
        print("Done.")
        
        if args.view:
            import subprocess
            print(f"Opening {args.output}...")
            try:
                if os.name == 'nt':
                    os.startfile(args.output)
                elif os.uname().sysname == 'Darwin':
                    subprocess.call(('open', args.output))
                else:
                    subprocess.call(('xdg-open', args.output))
            except Exception as e:
                print(f"Could not open file: {e}")

def save_gif(frames, probs, pred_tracks, gt_tracks, filename):
    import matplotlib.animation as animation
    
    T, H, W = frames.shape
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    def update(t):
        axes[0].clear()
        axes[1].clear()
        
        # Raw Image
        axes[0].imshow(np.log1p(frames[t]), cmap='inferno')
        axes[0].set_title(f"Frame {t} (Log Scale)")
        
        # Ground Truth annotations
        for gt in gt_tracks:
             pt = next((p for p in gt if p['t'] == t), None)
             if pt:
                 axes[0].plot(pt['x'], pt['y'], 'gx', markersize=10, markeredgewidth=2) # Green X
        
        # Prob Map
        axes[1].imshow(probs[t], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Model Probability")
        
        # Pred Track annotations
        for i, trk in enumerate(pred_tracks):
            pt = next((p for p in trk if p['t'] == t), None)
            if pt:
                axes[1].plot(pt['x'], pt['y'], 'ro', fillstyle='none', markersize=10, markeredgewidth=2) # Red Circle
                axes[1].text(pt['x']+2, pt['y']+2, f"T{i}", color='red')

    ani = animation.FuncAnimation(fig, update, frames=range(T), interval=200)
    ani.save(filename, writer='pillow')
    plt.close(fig)

if __name__ == "__main__":
    main()
