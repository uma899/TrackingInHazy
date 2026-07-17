import os
import cv2
import time
import torch
import argparse
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm

def pipeline_worker(gpu_id, dehaze_ckpt, track_ckpt, config_path, hazy_dir, frame_names, init_box, label, color, out_queue):
    """
    Worker function deployed to a specific GPU. 
    It reads frames, dehazes, tracks, draws bounding boxes, and pushes the frame to a queue.
    """
    # 1. Isolate this specific GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 2. Import agent inside the spawned process to ensure proper CUDA context binding
    from unified_wrapper import UnifiedDehazeTracker
    
    # Initialize the agent
    agent = UnifiedDehazeTracker(dehaze_ckpt, track_ckpt, config_path)
    
    # 3. Initialize the tracker on the first frame
    first_img_path = os.path.join(hazy_dir, frame_names[0])
    first_frame = cv2.imread(first_img_path)
    
    first_clean = agent.dehaze(first_frame)
    agent.init_tracker(first_clean, init_box)
    
    # 4. Process the remaining frames
    with torch.no_grad():
        for name in frame_names[1:]:
            path = os.path.join(hazy_dir, name)
            img = cv2.imread(path)
            if img is None:
                continue
                
            clean = agent.dehaze(img)
            res = agent.track_frame(clean)
            
            # Draw bbox and text
            bbox = list(map(int, res['bbox']))
            cv2.rectangle(clean, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)
            cv2.putText(clean, f"{label} ({res['best_score']:.2f})", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Send processed numpy array back to main process
            out_queue.put(clean)
            
    # Send termination signal
    out_queue.put(None)

def main():
    parser = argparse.ArgumentParser(description="Parallel Model Comparison Script (Standard vs Supervised)")
    
    # Standard Pipeline Paths
    parser.add_argument('--std_dehaze_ckpt', type=str, default='model_weights/model_80.pkl')
    parser.add_argument('--std_track_ckpt', type=str, default='model_weights/siamrpn_r50.pth')
    
    # Supervised Pipeline Paths
    parser.add_argument('--sup_dehaze_ckpt', type=str, default='checkpoints/eenet_model_80_pkl_supervised.pkl')
    parser.add_argument('--sup_track_ckpt', type=str, default='checkpoints/siamrpn_restored_finetuned_model_80.pth')
    
    # Global Config & IO
    parser.add_argument('--track_config', type=str, default='pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
    parser.add_argument('--hazy_dir', type=str, default='../IIT_HAZY/vid3/hazy')
    parser.add_argument('--out_video', type=str, default='out_model_80_comparision.mp4')
    
    # Execution Parameters
    parser.add_argument('--init_box', nargs=4, type=int, default=[698, 499, 248, 102], help='Initial bounding box: x y w h')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--gpu_std', type=int, default=0, help='GPU ID for standard model')
    parser.add_argument('--gpu_sup', type=int, default=1, help='GPU ID for supervised model')

    args = parser.parse_args()

    # Enforce spawn method for PyTorch multiprocessing
    mp.set_start_method('spawn', force=True)
    
    frame_names = sorted([f for f in os.listdir(args.hazy_dir) if f.endswith('.png')])
    if not frame_names:
        print(f"Error: No frames found in {args.hazy_dir}")
        return

    # Use queues with a max size to prevent RAM overflow if workers are faster than the video writer
    q_std = mp.Queue(maxsize=30)
    q_sup = mp.Queue(maxsize=30)

    print(f"Starting parallel processing...")
    print(f"-> Standard Pipeline on GPU {args.gpu_std}")
    print(f"-> Supervised Pipeline on GPU {args.gpu_sup}")

    # Spawn Worker Processes
    p_std = mp.Process(target=pipeline_worker, args=(
        args.gpu_std, args.std_dehaze_ckpt, args.std_track_ckpt, args.track_config, 
        args.hazy_dir, frame_names, args.init_box, "Standard", (0, 255, 0), q_std
    ))
    
    p_sup = mp.Process(target=pipeline_worker, args=(
        args.gpu_sup, args.sup_dehaze_ckpt, args.sup_track_ckpt, args.track_config, 
        args.hazy_dir, frame_names, args.init_box, "Supervised", (0, 0, 255), q_sup
    ))

    p_std.start()
    p_sup.start()

    # Setup Video Writer using the dimensions of the first frame
    first_frame = cv2.imread(os.path.join(args.hazy_dir, frame_names[0]))
    height, width, _ = first_frame.shape
    output_size = (width * 2, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.out_video, fourcc, args.fps, output_size)

    start_time = time.time()
    num_frames_to_process = len(frame_names) - 1

    # Read from the queues, stitch, and write synchronously
    for _ in tqdm(range(num_frames_to_process), desc="Stitching Final Video"):
        clean_std = q_std.get()
        clean_sup = q_sup.get()

        # If either worker failed or sent termination prematurely, break
        if clean_std is None or clean_sup is None:
            print("Received early termination signal from a worker.")
            break

        combined_frame = np.hstack((clean_std, clean_sup))
        out_video.write(combined_frame)

    # Cleanup
    p_std.join()
    p_sup.join()
    out_video.release()

    print(f"\nFinished parallel processing and video saved in {time.time() - start_time:.2f} seconds.")
    print(f"Output saved to: {args.out_video}")

if __name__ == "__main__":
    main()



"""
#coords = [698, 499, 248, 102]

import cv2
import os
import time
import torch
import numpy as np  
from tqdm import tqdm  # 1. IMPORT TQDM
from unified_wrapper import UnifiedDehazeTracker

torch.backends.cudnn.benchmark = True 

standard_agent = UnifiedDehazeTracker('model_weights/model_80.pkl',
    'model_weights/siamrpn_r50.pth', 
    'pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml'
)

super_agent = UnifiedDehazeTracker(
    'checkpoints/eenet_model_80_pkl_supervised.pkl', 
    'checkpoints/siamrpn_restored_finetuned_model_80.pth', 
    'pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml'
)

HAZY_PATH = '../IIT_HAZY/vid3/hazy'
frame_names = sorted([f for f in os.listdir(HAZY_PATH) if f.endswith('.png')])

# --- FIXED TRACKER INITIALIZATION ---
first_frame_hazy = cv2.imread(os.path.join(HAZY_PATH, frame_names[0]))
init_box = [698, 499, 248, 102] 

# Dehaze first frame so tracker templates match the cleaned search frames
first_frame_clean_std = standard_agent.dehaze(first_frame_hazy)
first_frame_clean_sup = super_agent.dehaze(first_frame_hazy)

standard_agent.init_tracker(first_frame_clean_std, init_box)
super_agent.init_tracker(first_frame_clean_sup, init_box)

height, width, layers = first_frame_hazy.shape
output_width = width * 2 
output_size = (output_width, height)
fps = 30  

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('out_model_80_comparision.mp4', fourcc, fps, output_size)

print("Comparing models and saving video (GPU Optimized)...")

start_time = time.time()
frame_paths = [os.path.join(HAZY_PATH, name) for name in frame_names[1:]]

# 2. WRAP ZIP IN TQDM AND REMOVE INDIVIDUAL PRINT STATEMENT
with torch.no_grad():
    for name, path in tqdm(zip(frame_names[1:], frame_paths), total=len(frame_paths), desc="Evaluating Frames"):
        h_img = cv2.imread(path)
        if h_img is None:
            continue
            
        # Run Standard Pipeline
        clean_std = standard_agent.dehaze(h_img)
        res_std = standard_agent.track_frame(clean_std)
        bbox_std = list(map(int, res_std['bbox'])) 
        cv2.rectangle(clean_std, (bbox_std[0], bbox_std[1]), (bbox_std[0]+bbox_std[2], bbox_std[1]+bbox_std[3]), (0, 255, 0), 2)
        cv2.putText(clean_std, f"Standard ({res_std['best_score']:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Run Supervised Pipeline
        clean_sup = super_agent.dehaze(h_img)
        res_sup = super_agent.track_frame(clean_sup)
        bbox_sup = list(map(int, res_sup['bbox']))
        cv2.rectangle(clean_sup, (bbox_sup[0], bbox_sup[1]), (bbox_sup[0]+bbox_sup[2], bbox_sup[1]+bbox_sup[3]), (0, 0, 255), 2)
        cv2.putText(clean_sup, f"Supervised ({res_sup['best_score']:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        combined_frame = np.hstack((clean_std, clean_sup))
        out_video.write(combined_frame)

out_video.release()

if torch.cuda.is_available():
    torch.cuda.synchronize()

print(f"\nFinished processing and video saved in {time.time() - start_time:.2f} seconds.")

"""