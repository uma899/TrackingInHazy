import os
import cv2
import time
import torch
import math
import argparse
import json
import multiprocessing as mp
from tqdm import tqdm
import subprocess
import shutil

def dehaze_worker(gpu_id, frames_chunk, dehaze_ckpt, track_ckpt, track_config, hazy_dir, temp_dir):
    """
    Worker function deployed to individual GPUs to handle the heavy dehazing.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    from unified_wrapper import UnifiedDehazeTracker
    agent = UnifiedDehazeTracker(dehaze_ckpt, track_ckpt, track_config)
    
    for name in tqdm(frames_chunk, position=gpu_id, desc=f"GPU {gpu_id} Dehazing"):
        img_path = os.path.join(hazy_dir, name)
        img = cv2.imread(img_path)
        
        if img is not None:
            with torch.no_grad():
                clean = agent.dehaze(img)
            cv2.imwrite(os.path.join(temp_dir, name), clean)

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Dehazing and Tracking Pipeline")
    
    parser.add_argument('--dehaze_ckpt', type=str, default='checkpoints/eenet_model_80_pkl_supervised.pkl')
    parser.add_argument('--track_ckpt', type=str, default='checkpoints/siamrpn_restored_finetuned_model_80.pth')
    parser.add_argument('--track_config', type=str, default='pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
    # parser.add_argument('--out_video', type=str, default='out_supervised_model_80.mp4', help='Path to output video file')
    # The flag defaults to False unless explicitly passed in the terminal
    parser.add_argument('--out_video', default=None, help='If set, saves the output video')


    parser.add_argument('--hazy_dir', type=str, default='../IIT_HAZY/vid3/hazy')
    #parser.add_argument('--temp_dir', type=str, default='./dehazed')
    parser.add_argument('--save_dehazed', action='store_true')
    
    parser.add_argument('--init_box', nargs=4, type=int, default=[698, 499, 248, 102])
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--start_gpu_id', type=int, default=0)
    
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--save_json', type=str, default='tracking_results.json', help='Path to save tracking results')
    parser.add_argument('--gt_json', type=str, default='tracking_results.json', help='Path to save tracking results')

    TEMP_DIR = './dehazed_frames'


    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    frame_names = sorted([f for f in os.listdir(args.hazy_dir) if f.endswith('.png')])
    
    if args.max_frames is not None:
        frame_names = frame_names[:args.max_frames]

    if not frame_names:
        print(f"Error: No PNG images found in {args.hazy_dir}")
        return

    chunk_size = math.ceil(len(frame_names) / args.num_gpus)
    chunks = [frame_names[i:i + chunk_size] for i in range(0, len(frame_names), chunk_size)]
    
    print(f"Phase 1: Starting Parallel Dehazing on {args.num_gpus} GPUs...")
    start_time = time.time()
    
    processes = []
    for i in range(min(args.num_gpus, len(chunks))):
        target_gpu = args.start_gpu_id + i
        p = mp.Process(
            target=dehaze_worker, 
            args=(target_gpu, chunks[i], args.dehaze_ckpt, args.track_ckpt, args.track_config, args.hazy_dir, TEMP_DIR)
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print(f"\nPhase 1 Complete! Dehazing took {time.time() - start_time:.2f} seconds.")
    
    # ==========================================================================
    # PHASE 2: Sequential Tracking (JSON Only)
    # ==========================================================================
    print("\nPhase 2: Initializing Tracker and Writing JSON...")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.start_gpu_id)
    from unified_wrapper import UnifiedDehazeTracker
    agent = UnifiedDehazeTracker(args.dehaze_ckpt, args.track_ckpt, args.track_config)
    
    first_frame_clean = cv2.imread(os.path.join(TEMP_DIR, frame_names[0]))
    agent.init_tracker(first_frame_clean, args.init_box)
    
    tracking_data = []
    
    tracking_data.append({
        "frame": frame_names[0],
        "bbox": [float(x) for x in args.init_box],
        "score": 1.0
    })

    for name in tqdm(frame_names[1:], desc="Tracking Only"):
        clean_img = cv2.imread(os.path.join(TEMP_DIR, name))
        if clean_img is None:
            continue
            
        with torch.no_grad():
            res = agent.track_frame(clean_img)
            
        tracking_data.append({
            "frame": name,
            "bbox": [float(x) for x in res['bbox']],
            "score": float(res['best_score'])
        })

    with open(args.save_json, 'w') as f:
        json.dump(tracking_data, f, indent=4)
        
    print(f"\nPipeline Complete! Tracking coordinates saved to {args.save_json}")
    
    command = [
            "python", "draw_box.py",
            "--json_path", args.save_json,
            # NOTE: TEMP_DIR now holds SCALE-resolution dehazed frames (the
            # dehazer only ever runs at SCALE resolution, matching training),
            # while tracking_data['bbox'] is upscaled back to ORIGINAL
            # resolution by UnifiedDehazeTracker.track_frame(). Drawing
            # original-resolution boxes onto SCALE-resolution frames puts the
            # box far outside the visible canvas -> nothing renders. Use
            # args.hazy_dir (original resolution, matches the JSON) instead.
            "--frames_dir", args.hazy_dir,
            "--output", args.out_video,
            "--gpus", "0,1,2,3"
        ]

    if args.out_video:
        try:
            print("\nVideo generation starts.. Make sure you have 4 GPUs")
            result = subprocess.run(command, check=True, text=True)

            if not args.save_dehazed:
                shutil.rmtree(TEMP_DIR)
                print("Process finished successfully! Deleting dehazed frames..")

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the script: {e}")    
    else:
        print("Video not saved. Run draw_box.py for that.")

    print("Comparision:")
    command2 = [
            "python", "results.py",
            "--gt_json", args.gt_json,
            "--pred_json", args.save_json,
            "--csv_out", "latest.csv",
            "--plot_out", "latest.png"
        ]    

    result2 = subprocess.run(command2, check=True, text=True)


if __name__ == "__main__":
    main()
