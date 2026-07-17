import os
import cv2
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool, Manager

# ==============================================================================
# CONFIGURATION & DEFAULT PATHS
# ==============================================================================
JSON_PATH        = 'gt_perfect_labels.json'
FRAMES_DIR       = '../IIT_HAZY/vid3/gt'
OUTPUT_VIDEO     = 'verify_json_boxes.mp4'
FPS              = 30
WORKERS_PER_GPU  = 2  # Boost this based on your CPU core count
EXTN = '.png'
# ==============================================================================

def process_chunk(args_tuple):
    """
    Highly isolated process worker. 
    More processes = faster disk reading and image decoding.
    """
    chunk_data, chunk_id, frames_dir, gpu_id, width, height, fps, progress_queue = args_tuple
    
    # Strictly bind this process to a specific GPU hardware instance
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    segment_filename = f"temp_chunk_{chunk_id}.mp4"
    
    # Fast encoding fallback. 
    # For true GPU encoding, ensure your OpenCV is compiled with CUDA/NVENC, 
    # or swap this out for a PyAV H264_NVENC container pipeline.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(segment_filename, fourcc, fps, (width, height))
    
    for item in chunk_data:
        frame_name = os.path.splitext(item['frame'])[0] + EXTN
        frame_path = os.path.join(frames_dir, frame_name)
        
        # CPU-heavy bottleneck occurs here (Image decoding)
        img = cv2.imread(frame_path)
        if img is not None:
            bx, by, bw, bh = map(int, item['bbox'])
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 0, 0), 3)
            cv2.putText(img, f"Score {item['score']}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            out_video.write(img)
            
        progress_queue.put(1)
        
    out_video.release()
    return segment_filename

def main():
    parser = argparse.ArgumentParser(description="Max Performance Parallel Video Verification")
    parser.add_argument('--json_path', type=str, default=JSON_PATH)
    parser.add_argument('--frames_dir', type=str, default=FRAMES_DIR)
    parser.add_argument('--output', type=str, default=OUTPUT_VIDEO)
    parser.add_argument('--gpus', type=str, default='0,1', help='Comma separated GPU IDs')
    parser.add_argument('--workers_per_gpu', type=int, default=WORKERS_PER_GPU, help='CPU threads/processes per GPU')
    parser.add_argument('--fps', type=int, default=FPS)
    args = parser.parse_args()

    print(f"Loading {args.json_path}...")
    with open(args.json_path, 'r') as f:
        data = json.load(f)
        
    if not data:
        return

    # Target first frame setup
    first_frame_name = os.path.splitext(data[0]['frame'])[0] + EXTN
    first_frame_path = os.path.join(args.frames_dir, first_frame_name)
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Error loading frame: {first_frame_path}")
        return
    height, width, _ = first_frame.shape

    # Calculate total parallel instances
    gpu_list = [int(g.strip()) for g in args.gpus.split(',')]
    total_workers = len(gpu_list) * args.workers_per_gpu
    print(f"Spawning {total_workers} total processes ({args.workers_per_gpu} per GPU across {len(gpu_list)} GPUs)")

    # Split dataset slice array evenly
    total_frames = len(data)
    chunk_size = (total_frames + total_workers - 1) // total_workers
    
    manager = Manager()
    progress_queue = manager.Queue()
    
    tasks = []
    for i in range(total_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_frames)
        chunk_data = data[start_idx:end_idx]
        
        # Map worker to a GPU using round-robin allocation
        assigned_gpu = gpu_list[i % len(gpu_list)]
        
        tasks.append((
            chunk_data, i, args.frames_dir, assigned_gpu, 
            width, height, args.fps, progress_queue
        ))

    # Run execution pool
    pool = Pool(processes=total_workers)
    pool_result = pool.map_async(process_chunk, tasks)

    with tqdm(total=total_frames, desc="Total Render Progress") as pbar:
        while not pool_result.ready():
            try:
                while not progress_queue.empty():
                    pbar.update(progress_queue.get_nowait())
            except:
                pass
            
    segment_files = pool_result.get()
    pool.close()
    pool.join()

    #Fast sequential stitcher
    print("\nStitching final video file...")
    final_video = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
    for seg_path in segment_files:
        if os.path.exists(seg_path):
            cap = cv2.VideoCapture(seg_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                final_video.write(frame)
            cap.release()
            os.remove(seg_path)

    final_video.release()
    print(f"Completed! Saved to: {args.output}")

if __name__ == "__main__":
    main()