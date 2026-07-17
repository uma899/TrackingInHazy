"""
Analyzes tracking results beyond the whole-video aggregate:

1. Confidence-gating trigger frequency: how many frames had score below
   conf_threshold (i.e. were held at the last confident position by
   unified_wrapper.py's gating fix), for a given prediction JSON. Run this
   on both the trained-weights and base-weights prediction JSONs to check
   whether one model is leaning on the safety net far more than the other -
   if so, its higher overall IoU may partly reflect "coasting" on held boxes
   rather than genuinely better tracking.

2. Segment-restricted metrics: the same style of metrics as your full-video
   summary (Mean IoU, Success Rate @ IoU>0.5, Success Plot AUC, Mean Center
   Error, Precision @ 20px, Precision Plot AUC, Full Failure Rate), but
   computed only over a specified frame range (e.g. the tree-background
   segment) instead of the whole video. This is the number that actually
   answers "did the fix help where it mattered," since a whole-video average
   can hide a segment-specific win or loss.

Usage:
    python analyze_results.py \
        --gt_json gt_labels.json \
        --pred_json results_joint_epoch2.json \
        --conf_threshold 0.3 \
        --segment_start 250 --segment_end 370

Run once per prediction JSON (trained, base) to compare.
"""
import json
import argparse
import numpy as np
import re


def _trapz(y, x):
    """Trapezoidal integration, avoiding numpy version differences
    (np.trapz was removed/renamed to np.trapezoid in newer numpy)."""
    y, x = np.asarray(y), np.asarray(x)
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * (x[1:] - x[:-1])))


def frame_key(name):
    """Extract the numeric frame index from a filename like '00000254.png'
    or '00000254.jpg', regardless of extension, for matching/filtering."""
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else None


def iou(box_a, box_b):
    """IoU for [x, y, w, h] boxes."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def center_error(box_a, box_b):
    ax, ay = box_a[0] + box_a[2] / 2.0, box_a[1] + box_a[3] / 2.0
    bx, by = box_b[0] + box_b[2] / 2.0, box_b[1] + box_b[3] / 2.0
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def compute_metrics(gt_by_frame, pred_by_frame, frame_ids):
    ious, center_errors = [], []
    for fid in frame_ids:
        if fid not in gt_by_frame or fid not in pred_by_frame:
            continue
        g, p = gt_by_frame[fid], pred_by_frame[fid]
        ious.append(iou(g, p))
        center_errors.append(center_error(g, p))

    if not ious:
        return None

    ious = np.array(ious)
    center_errors = np.array(center_errors)

    mean_iou = float(ious.mean())
    success_rate_50 = float((ious > 0.5).mean())

    # Success plot AUC: area under fraction-of-frames-above-threshold,
    # integrated over IoU thresholds 0..1. By construction this equals
    # mean IoU (E[IoU] = integral of P(IoU > t) dt for IoU in [0,1]), which
    # is why your full-video numbers show Success Plot AUC ~= Mean IoU.
    thresholds = np.linspace(0, 1, 101)
    success_curve = np.array([(ious > t).mean() for t in thresholds])
    success_auc = _trapz(success_curve, thresholds)

    mean_center_error = float(center_errors.mean())
    precision_20 = float((center_errors < 20).mean())

    # Precision plot AUC over 0-50px, normalized to 0-1 range by dividing
    # by the 50px window (matches how OTB-style precision AUC is usually
    # reported so it's comparable across ranges).
    px_thresholds = np.linspace(0, 50, 101)
    precision_curve = np.array([(center_errors < t).mean() for t in px_thresholds])
    precision_auc = _trapz(precision_curve, px_thresholds) / 50.0

    full_failure_rate = float((ious <= 0.0).mean())

    return {
        'frames_evaluated': len(ious),
        'mean_iou': mean_iou,
        'success_rate_50': success_rate_50,
        'success_auc': success_auc,
        'mean_center_error': mean_center_error,
        'precision_20': precision_20,
        'precision_auc': precision_auc,
        'full_failure_rate': full_failure_rate,
    }


def print_metrics(title, m):
    print("=" * 60)
    print(title)
    print("=" * 60)
    if m is None:
        print("No overlapping frames found in this range.")
        return
    print(f"Frames evaluated:                {m['frames_evaluated']}")
    print(f"Mean IoU:                        {m['mean_iou']:.4f}")
    print(f"Success Rate @ IoU > 0.50:        {m['success_rate_50']:.4f}")
    print(f"Success Plot AUC (0-1):           {m['success_auc']:.4f}")
    print(f"Mean Center Error (px):           {m['mean_center_error']:.2f}")
    print(f"Precision @ 20px:                 {m['precision_20']:.4f}")
    print(f"Precision Plot AUC (0-50px):      {m['precision_auc']:.4f}")
    print(f"Full Failure Rate (IoU <= 0.00):  {m['full_failure_rate']:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json', type=str, required=True)
    parser.add_argument('--pred_json', type=str, required=True)
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                         help='Must match the conf_threshold used in unified_wrapper.py track_frame()')
    parser.add_argument('--segment_start', type=int, default=None,
                         help='Frame index (parsed from filename) where the segment of interest starts')
    parser.add_argument('--segment_end', type=int, default=None,
                         help='Frame index where the segment of interest ends (inclusive)')
    args = parser.parse_args()

    with open(args.gt_json, 'r') as f:
        gt_data = json.load(f)
    with open(args.pred_json, 'r') as f:
        pred_data = json.load(f)

    gt_by_frame = {frame_key(item['frame']): item['bbox'] for item in gt_data}
    pred_by_frame = {frame_key(item['frame']): item['bbox'] for item in pred_data}
    pred_score_by_frame = {frame_key(item['frame']): item.get('score', None) for item in pred_data}

    all_frame_ids = sorted(set(gt_by_frame.keys()) & set(pred_by_frame.keys()))

    # ------------------------------------------------------------------
    # 1. Confidence-gating trigger frequency
    # ------------------------------------------------------------------
    scores = [s for s in pred_score_by_frame.values() if s is not None]
    held_count = sum(1 for s in scores if s < args.conf_threshold)
    print(f"\n--- Confidence-gating summary ({args.pred_json}) ---")
    print(f"Frames with score < {args.conf_threshold} (held/frozen position): "
          f"{held_count} / {len(scores)} ({100.0 * held_count / max(1, len(scores)):.1f}%)")

    if args.segment_start is not None and args.segment_end is not None:
        seg_scores = [pred_score_by_frame[fid] for fid in all_frame_ids
                      if args.segment_start <= fid <= args.segment_end
                      and pred_score_by_frame.get(fid) is not None]
        seg_held = sum(1 for s in seg_scores if s < args.conf_threshold)
        print(f"Within segment [{args.segment_start}, {args.segment_end}]: "
              f"{seg_held} / {len(seg_scores)} held "
              f"({100.0 * seg_held / max(1, len(seg_scores)):.1f}%)")

    # ------------------------------------------------------------------
    # 2. Whole-video metrics (sanity check against your existing numbers)
    # ------------------------------------------------------------------
    full_metrics = compute_metrics(gt_by_frame, pred_by_frame, all_frame_ids)
    print_metrics(f"\nWHOLE-VIDEO metrics ({args.pred_json})", full_metrics)

    # ------------------------------------------------------------------
    # 3. Segment-restricted metrics
    # ------------------------------------------------------------------
    if args.segment_start is not None and args.segment_end is not None:
        segment_ids = [fid for fid in all_frame_ids
                        if args.segment_start <= fid <= args.segment_end]
        segment_metrics = compute_metrics(gt_by_frame, pred_by_frame, segment_ids)
        print_metrics(
            f"SEGMENT [{args.segment_start}-{args.segment_end}] metrics ({args.pred_json})",
            segment_metrics
        )


if __name__ == "__main__":
    main()