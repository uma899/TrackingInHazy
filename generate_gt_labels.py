import cv2
import os
import json
import argparse
from unified_wrapper import UnifiedDehazeTracker

def main():
    parser = argparse.ArgumentParser(description="Generate tracking labels from Ground Truth frames")
    
    # Configuration & Paths
    parser.add_argument('--gt_dir', type=str, default='../IIT_HAZY/vid3/gt', help='Directory containing ground truth frames (.jpg)')
    parser.add_argument('--dehaze_ckpt', type=str, default='checkpoints/eenet_ots.pkl', help='Path to dehaze model (required by wrapper)')
    parser.add_argument('--track_ckpt', type=str, default='checkpoints/siamrpn_r50.pth', help='Path to tracker weights')
    parser.add_argument('--track_config', type=str, default='pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml', help='Path to tracker config file')
    parser.add_argument('--out_json', type=str, default='gt_perfect_labels.json', help='Path to save output JSON file')
    
    # Tracker Initial State
    parser.add_argument('--init_box', nargs=4, type=int, default=[698, 499, 248, 102], help='Initial bounding box: x y w h (Space separated)')

    args = parser.parse_args()

    # Initialize Agent
    agent = UnifiedDehazeTracker(args.dehaze_ckpt, args.track_ckpt, args.track_config)
    frame_names = sorted([f for f in os.listdir(args.gt_dir) if f.endswith('.png')])

    if not frame_names:
        print(f"Error: No JPG images found in {args.gt_dir}")
        return

    # Initialize Tracker on the first frame
    first_frame = cv2.imread(os.path.join(args.gt_dir, frame_names[0]))
    agent.init_tracker(first_frame, args.init_box)

    gt_manifest = []
    print(f"Generating perfect labels from {args.gt_dir}...")

    # Tracking Loop
    for name in frame_names:
        img = cv2.imread(os.path.join(args.gt_dir, name))
        if img is None:
            print(f"Warning: Could not read {name}, skipping.")
            continue
            
        out = agent.track_frame(img)
        gt_manifest.append({
            "frame": name,
            "bbox": [float(x) for x in out['bbox']],
            "score": float(out['best_score'])
        })
        print(name, ":", gt_manifest[-1])

    # Ensure output directory exists if saving to a subfolder
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save to JSON
    with open(args.out_json, 'w') as f:
        json.dump(gt_manifest, f, indent=4)
        
    print(f"Done! Created {args.out_json}")

if __name__ == "__main__":
    main()
    

"""
import cv2
import os
import json
from unified_wrapper import UnifiedDehazeTracker

# CONFIG
GT_PATH = '../IIT_HAZY/vid3/gt'
TRACK_MODEL = 'checkpoints/siamrpn_r50.pth'
TRACK_CONFIG = 'pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml'

agent = UnifiedDehazeTracker('checkpoints/eenet_ots.pkl', TRACK_MODEL, TRACK_CONFIG)
frame_names = sorted([f for f in os.listdir(GT_PATH) if f.endswith('.jpg')])

first_frame = cv2.imread(os.path.join(GT_PATH, frame_names[0]))
init_box = [698, 499, 248, 102]
agent.init_tracker(first_frame, list(init_box))

gt_manifest = []
print("Generating perfect labels from GT folder...")

for name in frame_names:
    img = cv2.imread(os.path.join(GT_PATH, name))
    out = agent.track_frame(img)
    gt_manifest.append({
        "frame": name,
        "bbox": [float(x) for x in out['bbox']],
        "score": float(out['best_score'])
    })
    print(name, ":", gt_manifest[-1])

with open('gt_perfect_labels.json', 'w') as f:
    json.dump(gt_manifest, f, indent=4)
print("Done! Created gt_perfect_labels.json")
"""