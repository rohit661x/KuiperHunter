import pytest
import numpy as np
from src.evaluation.tracker import SimpleLinearTracker
from src.evaluation.metrics import compute_metrics, boxes_to_discrete_tracks

def test_tracker_perfect_line():
    # Synthetic perfect track
    dets = []
    # x = t + 10, y = 10
    for t in range(5):
        dets.append({'t': t, 'x': 10.0 + t, 'y': 10.0, 'score': 1.0})
        
    tracker = SimpleLinearTracker(max_velocity=2.0, min_track_len=5)
    tracks = tracker.link_detections(dets)
    
    assert len(tracks) == 1
    assert len(tracks[0]) == 5
    assert tracks[0][0]['t'] == 0
    assert tracks[0][-1]['t'] == 4

def test_tracker_gap_handling():
    # Line with a missing frame at t=2
    dets = []
    for t in [0, 1, 3, 4]:
        dets.append({'t': t, 'x': 10.0 + t, 'y': 10.0, 'score': 1.0})
        
    tracker = SimpleLinearTracker(max_velocity=2.0, min_track_len=4, missed_frames=1)
    tracks = tracker.link_detections(dets)
    
    assert len(tracks) == 1
    assert len(tracks[0]) == 4 # Should recover the track skipping t=2

def test_metrics_exact_match():
    # Pred = GT
    track = [{'t': t, 'x': t, 'y': t} for t in range(5)]
    pred_tracks = [track]
    gt_tracks = [track]
    
    m = compute_metrics(pred_tracks, gt_tracks, distance_threshold=1.0)
    assert m['tp'] == 1
    assert m['fp'] == 0
    assert m['fn'] == 0
    assert m['precision'] == 1.0
    assert m['recall'] == 1.0

def test_boxes_to_track_conversion():
    movers = [{
        'x0': 0, 'y0': 0, 'vx': 1, 'vy': 0, 
        'flux': 100, 'mag': 20, 'id': 0
    }]
    T = 3
    tracks = boxes_to_discrete_tracks(movers, T)
    
    assert len(tracks) == 1
    t0 = tracks[0][0]
    t2 = tracks[0][2]
    
    assert t0['x'] == 0
    assert t2['x'] == 2 # 0 + 1*2
