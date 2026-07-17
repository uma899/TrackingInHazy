# track_hazy.py
import sys
import os
import cv2
import time
import torch
import argparse
from tqdm import tqdm
import json

# --- FIX: ADD PATHS SO PYTHON CAN SEE PYSOT ---
# This assumes you are running the script from the 'Main' directory
sys.path.append(os.path.join(os.getcwd(), 'pysot'))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
# ----------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline Tracking directly on Hazy frames (No Dehazing)")
    
    # Tracker Config (Defaulting to the base model, but you can pass your fine-tuned one)
    parser.add_argument('--track_ckpt', type=str, default='model_weights/siamrpn_r50.pth', help='Path to tracker weights')
    parser.add_argument('--track_config', type=str, default='pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml', help='Path to tracker config file')
    
    # IO Paths
    parser.add_argument('--hazy_dir', type=str, default='../IIT_HAZY/vid3/hazy', help='Directory containing input hazy frames (PNGs)')
    parser.add_argument('--out_video', type=str, default='out_baseline_hazy_tracking.mp4', help='Path to output video file')
    
    # Execution Parameters
    parser.add_argument('--init_box', nargs=4, type=int, default=[698, 499, 248, 102], help='Initial bounding box: x y w h (Space separated)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for output video')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process (None for all)')
    parser.add_argument('--save_json', type=str, default=None, help='Path to save tracking results as JSON (e.g., tracking_results.json)')

    args = parser.parse_args()

    # 1. Setup Environment
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    print("Loading PySOT SiamRPN Tracker...")
    cfg.merge_from_file(args.track_config)
    
    model = ModelBuilder()
    model.load_state_dict(torch.load(args.track_ckpt, map_location='cpu'))
    model.eval().cuda()
    
    tracker = build_tracker(model)

    
    # 2. Load Frames
    frame_names = sorted([f for f in os.listdir(args.hazy_dir) if f.endswith('.png')])
    if not frame_names:
        print(f"Error: No PNG images found in {args.hazy_dir}")
        return
        
    # --- NEW CODE: Slice the list if max_frames is provided ---
    if args.max_frames is not None:
        frame_names = frame_names[:args.max_frames]
        print(f"Limiting execution to {args.max_frames} frames...")
    # ----------------------------------------------------------    

    # 3. Setup Video Writer
    first_frame = cv2.imread(os.path.join(args.hazy_dir, frame_names[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.out_video, fourcc, args.fps, (width, height))

    # 4. Initialize Tracker on First Frame
    print(f"Initializing tracker with box: {args.init_box}")
    tracker.init(first_frame, args.init_box)
    
    # Draw initial box (Green)
    start_img = first_frame.copy()
    x, y, w, h = args.init_box
    cv2.rectangle(start_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(start_img, "Init (Baseline)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out_video.write(start_img)

    # 5. Tracking Loop (Sequential)
    print("Starting tracking on hazy frames...")
    start_time = time.time()

    tracking_data = []

    for name in tqdm(frame_names[1:], desc="Tracking"):
        img_path = os.path.join(args.hazy_dir, name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        # Track directly on hazy image
        with torch.no_grad():
            outputs = tracker.track(img)
            
        bbox = list(map(int, outputs['bbox']))
        score = outputs.get('best_score', 0.0)
        
        # --- ADD THIS BLOCK to log the exact format you want ---
        tracking_data.append({
            "frame": name,
            "bbox": [float(x) for x in outputs['bbox']], # Storing the precise floats
            "score": float(score)
        })
        # -------------------------------------------------------

        # Draw tracked box (Red)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 3)
        cv2.putText(img, f"Tracked ({score:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out_video.write(img)

    out_video.release()

    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(tracking_data, f, indent=4)
        print(f"Tracking coordinates saved to {args.save_json}")

    print(f"\nBaseline tracking complete in {time.time() - start_time:.2f} seconds!")
    print(f"Video saved to {args.out_video}")

if __name__ == "__main__":
    main()