--- Confidence-gating summary (hazy_pred_labels_vid3.json) ---
Frames with score < 0.3 (held/frozen position): 123 / 2428 (5.1%)
Within segment [250, 370]: 112 / 121 held (92.6%)
============================================================

WHOLE-VIDEO metrics (hazy_pred_labels_vid3.json)
============================================================
Frames evaluated:                2428
Mean IoU:                        0.8235
Success Rate @ IoU > 0.50:        0.9535
Success Plot AUC (0-1):           0.8236
Mean Center Error (px):           6.92
Precision @ 20px:                 0.9481
Precision Plot AUC (0-50px):      0.8720
Full Failure Rate (IoU <= 0.00):  0.0000
============================================================
============================================================
SEGMENT [250-370] metrics (hazy_pred_labels_vid3.json)
============================================================
Frames evaluated:                121
Mean IoU:                        0.3700
Success Rate @ IoU > 0.50:        0.1818
Success Plot AUC (0-1):           0.3704
Mean Center Error (px):           47.46
Precision @ 20px:                 0.0909
Precision Plot AUC (0-50px):      0.2144
Full Failure Rate (IoU <= 0.00):  0.0000
============================================================

--- Confidence-gating summary (hazy_baseline_results_vid3.json) ---
Frames with score < 0.3 (held/frozen position): 95 / 2428 (3.9%)
Within segment [250, 370]: 95 / 121 held (78.5%)
============================================================

WHOLE-VIDEO metrics (hazy_baseline_results_vid3.json)
============================================================
Frames evaluated:                2428
Mean IoU:                        0.7668
Success Rate @ IoU > 0.50:        0.9580
Success Plot AUC (0-1):           0.7668
Mean Center Error (px):           6.61
Precision @ 20px:                 0.9633
Precision Plot AUC (0-50px):      0.8807
Full Failure Rate (IoU <= 0.00):  0.0000
============================================================
============================================================
SEGMENT [250-370] metrics (hazy_baseline_results_vid3.json)
============================================================
Frames evaluated:                121
Mean IoU:                        0.4236
Success Rate @ IoU > 0.50:        0.2645
Success Plot AUC (0-1):           0.4236
Mean Center Error (px):           45.44
Precision @ 20px:                 0.3058
Precision Plot AUC (0-50px):      0.3498
Full Failure Rate (IoU <= 0.00):  0.0000
============================================================



Training Output last:
Epoch 5 | Iter 560 | cls=0.0000 loc=0.1207 track=0.1207 pixel_raw=7.6094 pixel_weighted=0.2283 total=0.3490
Epoch 5 | Iter 570 | cls=0.0001 loc=0.0827 track=0.0828 pixel_raw=8.7283 pixel_weighted=0.2618 total=0.3446
Epoch 5 | Iter 580 | cls=0.0000 loc=0.1219 track=0.1220 pixel_raw=10.4452 pixel_weighted=0.3134 total=0.4353
Epoch 5 | Iter 590 | cls=0.0003 loc=0.1816 track=0.1819 pixel_raw=8.3741 pixel_weighted=0.2512 total=0.4332
Saved epoch 5 checkpoints.



