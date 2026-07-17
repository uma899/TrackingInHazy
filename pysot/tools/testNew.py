import argparse
import os
import glob
import cv2
import torch
import numpy as np

# PySOT imports
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

parser = argparse.ArgumentParser(description='PySOT Colab Headless Tracker')
parser.add_argument('--config', type=str, required=True, help='config file path')
parser.add_argument('--snapshot', type=str, required=True, help='model snapshot path')
parser.add_argument('--video', type=str, required=True, help='path to folder containing video frames')
parser.add_argument('--output', type=str, default='tracked_output.mp4', help='output video path')
args = parser.parse_args()

def main():
    # Load configuration
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # Build model and tracker
    model = ModelBuilder()
    model.load_state_dict(torch.load(args.snapshot, map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)
    tracker = build_tracker(model)

    # Read image frames from the directory
    extensions = ('*.jpg', '*.jpeg', '*.png')
    frames_list = []
    for ext in extensions:
        frames_list.extend(glob.glob(os.path.join(args.video, ext)))
    frames_list.sort() # Ensure correct sequential video tracking order

    if len(frames_list) == 0:
        print(f"Error: No image frames found in {args.video}")
        return

    print(f"Found {len(frames_list)} frames. Starting processing...")

    # Load first frame to initialize tracker and video writer
    first_frame = cv2.imread(frames_list[0])
    height, width, _ = first_frame.shape

    # Set up headless output video encoder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 30.0, (width, height))

    # --- SIMULATE SELECTION ON FRAME 1 ---
    # Since we can't draw a box with a mouse on Colab, we default to tracking the center 20% area.
    # Replace these coordinates with your actual target coordinates if known [xmin, ymin, width, height]
    
    
    
    #init_rect = [int(width * 0.4), int(height * 0.4), int(width * 0.2), int(height * 0.2)]
    init_rect = [576, 547, 912, 480]
    
    # Initialize tracker on Frame 1
    tracker.init(first_frame, init_rect)
    
    # Draw initial box and write frame
    cv2.rectangle(first_frame, (init_rect[0], init_rect[1]), 
                  (init_rect[0]+init_rect[2], init_rect[1]+init_rect[3]), (0, 255, 0), 3)
    out.write(first_frame)

    # --- TRACKING LOOP THROUGH REMAINING FRAMES ---
    for idx, img_path in enumerate(frames_list[1:], start=2):
        frame = cv2.imread(img_path)
        
        # Core tracker matching calculation update
        outputs = tracker.track(frame)
        
        # Parse tracking bounding box prediction results
        bbox = list(map(int, outputs['bbox']))
        
        # Draw the target box predictions on the current frame
        cv2.rectangle(frame, (bbox[0], bbox[1]), 
                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 3)
        
        # Overlay a progress stamp text
        cv2.putText(frame, f"Tracking Frame: {idx}/{len(frames_list)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Commit updated frame into our saved output file
        out.write(frame)
        
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(frames_list)} frames...")

    out.release()
    print(f"Tracking completely finished! Saved file to: {args.output}")

if __name__ == '__main__':
    main()