"""
Debug script: prints the RAW (pre-clip) predicted box each frame, by
monkey-patching PySOT's tracker._bbox_clip so we can see what the
regression head actually output before it gets clamped to frame bounds.

This is intentionally cheap: it only tracks the first N frames and reuses
already-dehazed frames in TEMP_DIR if present (falls back to dehazing them
fresh if not).

Usage (mirrors eval_parallel.py's relevant args):
    python debug_track_raw.py \
        --dehaze_ckpt checkpoints/eenet_model_80_pkl_supervised.pkl \
        --track_ckpt checkpoints/siamrpn_restored_finetuned_model_80.pth \
        --track_config pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
        --hazy_dir ../IIT_HAZY/vid3/hazy \
        --init_box 698 499 248 102 \
        --num_frames 15
"""
import os
import cv2
import torch
import argparse
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dehaze_ckpt', type=str, default='checkpoints/eenet_model_80_pkl_supervised.pkl')
    parser.add_argument('--track_ckpt', type=str, default='checkpoints/siamrpn_restored_finetuned_model_80.pth')
    parser.add_argument('--track_config', type=str, default='pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
    parser.add_argument('--hazy_dir', type=str, default='../IIT_HAZY/vid3/hazy')
    parser.add_argument('--temp_dir', type=str, default='./dehazed_frames', help='Reuses frames here if present')
    parser.add_argument('--init_box', nargs=4, type=int, default=[698, 499, 248, 102])
    parser.add_argument('--num_frames', type=int, default=15)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--reuse_existing', action='store_true',
                         help='Reuse already-dehazed frames in temp_dir if present. '
                              'OFF by default: stale frames left over from a DIFFERENT '
                              'checkpoint or an earlier run would otherwise be silently '
                              'reused, making results look fine when they are not '
                              'actually testing the checkpoint you specified.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    from unified_wrapper import UnifiedDehazeTracker
    agent = UnifiedDehazeTracker(args.dehaze_ckpt, args.track_ckpt, args.track_config)

    # ------------------------------------------------------------------
    # Monkey-patch _bbox_clip on the underlying PySOT tracker instance to
    # log its inputs (the raw, pre-clip cx, cy, width, height) before
    # calling through to the original implementation.
    # ------------------------------------------------------------------
    raw_predictions = []
    original_bbox_clip = agent.tracker._bbox_clip

    def logged_bbox_clip(self, cx, cy, width, height, boundary):
        raw_predictions.append((cx, cy, width, height))
        return original_bbox_clip(cx, cy, width, height, boundary)

    agent.tracker._bbox_clip = types.MethodType(logged_bbox_clip, agent.tracker)

    # ------------------------------------------------------------------
    # Get frame names and dehaze (or reuse) the first num_frames+1 frames
    # ------------------------------------------------------------------
    frame_names = sorted([f for f in os.listdir(args.hazy_dir) if f.endswith('.png')])[:args.num_frames + 1]

    os.makedirs(args.temp_dir, exist_ok=True)
    for name in frame_names:
        out_path = os.path.join(args.temp_dir, name)
        if args.reuse_existing and os.path.exists(out_path):
            continue
        img = cv2.imread(os.path.join(args.hazy_dir, name))
        if img is None:
            continue
        with torch.no_grad():
            clean = agent.dehaze(img)
        cv2.imwrite(out_path, clean)

    # ------------------------------------------------------------------
    # Init on frame 0, track through the rest, logging raw + clipped bbox
    # ------------------------------------------------------------------
    first_frame_clean = cv2.imread(os.path.join(args.temp_dir, frame_names[0]))
    agent.init_tracker(first_frame_clean, args.init_box)

    print(f"{'frame':>14} | {'raw cx,cy,w,h (pre-clip)':>45} | {'clipped bbox (orig-res)':>35} | score")
    print("-" * 130)

    for name in frame_names[1:]:
        clean_img = cv2.imread(os.path.join(args.temp_dir, name))
        if clean_img is None:
            continue
        with torch.no_grad():
            res = agent.track_frame(clean_img)

        raw = raw_predictions[-1] if raw_predictions else None
        raw_str = f"{raw}" if raw is not None else "N/A (patch didn't fire)"
        print(f"{name:>14} | {raw_str:>45} | {str([round(c, 1) for c in res['bbox']]):>35} | {res['best_score']:.4f}")


if __name__ == "__main__":
    main()
