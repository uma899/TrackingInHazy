"""
Compare a predicted tracking-output JSON against a ground-truth JSON, both in
the format:

    {
        "frame": "00000001.png",
        "bbox": [x, y, w, h],
        "score": 0.988884687423706
    }

Computes standard single-object-tracking metrics:
  - Per-frame IoU (Intersection over Union)
  - Mean IoU
  - Success Rate @ IoU threshold(s)
  - Success Plot AUC (mean success rate integrated over IoU thresholds 0..1,
    the standard OTB/VOT "success score")
  - Center Location Error (CLE, Euclidean distance between box centers)
  - Precision @ CLE threshold (default 20px, standard OTB convention)

Usage:
    python compare_tracking_accuracy.py \
        --gt_json gt_perfect_labels.json \
        --pred_json output_tracks.json \
        --iou_threshold 0.5 \
        --center_threshold 20 \
        --csv_out per_frame_results.csv \
        --plot_out success_plot.png
"""
import os
import json
import argparse
import csv
import numpy as np

# numpy >=2.0 renamed trapz -> trapezoid; support both
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def parse_args():
    parser = argparse.ArgumentParser(description="Compare tracking accuracy against GT manifest")
    parser.add_argument('--gt_json', type=str, required=True,
                        help='Path to ground-truth JSON (list of {frame, bbox, score})')
    parser.add_argument('--pred_json', type=str, required=True,
                        help='Path to predicted/output JSON, same format')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for Success Rate reporting (default: 0.5)')
    parser.add_argument('--center_threshold', type=float, default=20.0,
                        help='Center Location Error threshold in pixels for Precision reporting (default: 20, OTB standard)')
    parser.add_argument('--csv_out', type=str, default=None,
                        help='Optional path to write per-frame IoU/CLE results as CSV')
    parser.add_argument('--plot_out', type=str, default=None,
                        help='Optional path to save a success-plot PNG (requires matplotlib)')
    parser.add_argument('--fail_iou', type=float, default=0.0,
                        help='Frames with IoU <= this value are counted as full tracking failures (default: 0.0)')
    return parser.parse_args()


def load_manifest(path):
    with open(path, 'r') as f:
        data = json.load(f)
    # Key by extension-stripped basename so GT (.jpg) can match
    # predictions (.png) for the same underlying frame index.
    manifest = {}
    for item in data:
        base_name = os.path.splitext(item['frame'])[0]
        if base_name in manifest:
            print(f"WARNING: duplicate frame key '{base_name}' in {path} "
                  f"(from '{manifest[base_name]['frame']}' and '{item['frame']}') - keeping the last one.")
        manifest[base_name] = item
    return manifest


def compute_iou(box_a, box_b):
    """box: [x, y, w, h] -> IoU in [0, 1]"""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def compute_center_error(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    acx, acy = ax1 + aw / 2.0, ay1 + ah / 2.0
    bcx, bcy = bx1 + bw / 2.0, by1 + bh / 2.0
    return float(np.hypot(acx - bcx, acy - bcy))


def main():
    args = parse_args()

    gt = load_manifest(args.gt_json)
    pred = load_manifest(args.pred_json)

    gt_frames = set(gt.keys())
    pred_frames = set(pred.keys())
    common_frames = sorted(gt_frames & pred_frames, key=lambda n: n)

    missing_in_pred = gt_frames - pred_frames
    missing_in_gt = pred_frames - gt_frames

    if missing_in_pred:
        print(f"WARNING: {len(missing_in_pred)} frame(s) in GT are missing from predictions "
              f"(excluded from metrics). Example: {sorted(missing_in_pred)[:3]}")
    if missing_in_gt:
        print(f"NOTE: {len(missing_in_gt)} frame(s) in predictions are not in GT (ignored).")

    if not common_frames:
        print("ERROR: No overlapping frames between GT and predictions - nothing to compare.")
        return

    ious = []
    center_errors = []
    per_frame_rows = []

    for name in common_frames:
        gt_box = gt[name]['bbox']
        pred_box = pred[name]['bbox']

        iou = compute_iou(gt_box, pred_box)
        cle = compute_center_error(gt_box, pred_box)

        ious.append(iou)
        center_errors.append(cle)
        per_frame_rows.append({
            'frame': name,
            'iou': iou,
            'center_error_px': cle,
            'gt_bbox': gt_box,
            'pred_bbox': pred_box,
            'pred_score': pred[name].get('score', None),
        })

    ious = np.array(ious)
    center_errors = np.array(center_errors)

    mean_iou = float(np.mean(ious))
    success_rate_at_thresh = float(np.mean(ious > args.iou_threshold))
    precision_at_thresh = float(np.mean(center_errors <= args.center_threshold))
    failure_rate = float(np.mean(ious <= args.fail_iou))

    # Success Plot AUC: mean success rate integrated over IoU thresholds 0..1
    # (standard OTB "success score" / overlap-precision AUC)
    thresholds = np.linspace(0, 1, 101)
    success_curve = np.array([np.mean(ious > t) for t in thresholds])
    success_auc = float(_trapz(success_curve, thresholds))

    # Precision Plot AUC: mean precision integrated over CLE thresholds 0..50px
    cle_thresholds = np.linspace(0, 50, 101)
    precision_curve = np.array([np.mean(center_errors <= t) for t in cle_thresholds])
    precision_auc = float(_trapz(precision_curve, cle_thresholds) / 50.0)  # normalized to [0,1]

    print("\n" + "=" * 60)
    print(f"Frames evaluated: {len(common_frames)} / {len(gt_frames)} GT frames")
    print("=" * 60)
    print(f"Mean IoU:                      {mean_iou:.4f}")
    print(f"Success Rate @ IoU > {args.iou_threshold:.2f}:      {success_rate_at_thresh:.4f}")
    print(f"Success Plot AUC (0-1):        {success_auc:.4f}")
    print(f"Mean Center Error (px):        {np.mean(center_errors):.2f}")
    print(f"Precision @ {args.center_threshold:.0f}px:            {precision_at_thresh:.4f}")
    print(f"Precision Plot AUC (0-50px):   {precision_auc:.4f}")
    print(f"Full Failure Rate (IoU <= {args.fail_iou:.2f}): {failure_rate:.4f}")
    print("=" * 60)

    if args.csv_out:
        with open(args.csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'iou', 'center_error_px', 'gt_bbox', 'pred_bbox', 'pred_score'])
            writer.writeheader()
            for row in per_frame_rows:
                writer.writerow(row)
        print(f"Per-frame results written to: {args.csv_out}")

    if args.plot_out:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].plot(thresholds, success_curve, linewidth=2)
            axes[0].set_xlabel('IoU threshold')
            axes[0].set_ylabel('Success Rate')
            axes[0].set_title(f'Success Plot (AUC = {success_auc:.3f})')
            axes[0].set_xlim(0, 1)
            axes[0].set_ylim(0, 1)
            axes[0].grid(alpha=0.3)

            axes[1].plot(cle_thresholds, precision_curve, linewidth=2, color='darkorange')
            axes[1].set_xlabel('Center Error threshold (px)')
            axes[1].set_ylabel('Precision')
            axes[1].set_title(f'Precision Plot (AUC = {precision_auc:.3f})')
            axes[1].set_xlim(0, 50)
            axes[1].set_ylim(0, 1)
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.plot_out, dpi=150)
            print(f"Plot saved to: {args.plot_out}")
        except ImportError:
            print("matplotlib not available - skipping --plot_out (metrics above are still valid).")


if __name__ == "__main__":
    main()