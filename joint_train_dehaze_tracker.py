"""
TRUE joint/co-training of the dehazer + tracker RPN head.

Unlike cotrain_finetune_tracker.py (which finetunes the tracker on FROZEN,
pre-exported dehazed frames - a one-way, sequential handoff), this script
puts both networks in a single differentiable pipeline:

    raw hazy frame -> dehazer (trainable) -> differentiable crop
                    -> tracker RPN head (trainable) -> cls/loc loss
                    -> backprop through BOTH networks together

A pixel loss (dehazer output vs GT, on the same crop) is combined with the
tracking loss so the dehazer can't drift into "adversarial" pixels that fool
the tracker's confidence without producing a real, sensible dehazed image.

This is significantly more expensive per-iteration than the decoupled
approach (2 dehazer forward passes per training sample, every step, instead
of dehazing once ahead of time) - expect slower iterations.

EXPERIMENTAL / UNTESTED: I don't have your EENet/ModelBuilder source to run
this against, so verify the very first few printed loss values (not NaN,
not huge) before trusting a long run. See the caveats printed at the end
of this file's usage instructions.
"""
import sys
import os
import argparse
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

sys.path.append(os.path.join(os.getcwd(), 'EENet/OTS'))
sys.path.append(os.path.join(os.getcwd(), 'pysot'))

from models.EENet import build_net as build_eenet
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg
from pysot.datasets.anchor_target import AnchorTarget

CONTEXT_AMOUNT = 0.5
EXEMPLAR_SIZE = 127
SEARCH_SIZE = 255


# ==============================================================================
# Dataset: returns RAW hazy + GT frames (numpy), not pre-dehazed. Dehazing
# happens inside the training loop, on GPU, with gradients enabled.
# ==============================================================================
class JointDataset(Dataset):
    def __init__(self, json_path, hazy_dir, gt_dir, max_frames=None):
        with open(json_path, 'r') as f:
            full_data = json.load(f)

        existing = [
            item for item in full_data
            if os.path.exists(os.path.join(hazy_dir, os.path.splitext(item['frame'])[0] + '.png'))
            and os.path.exists(os.path.join(gt_dir, os.path.splitext(item['frame'])[0] + '.jpg'))
        ]
        missing = len(full_data) - len(existing)
        if missing:
            print(f"[{json_path}] {missing} / {len(full_data)} frames missing hazy or GT image - excluded.")

        if max_frames is not None and max_frames < len(existing):
            idx = np.linspace(0, len(existing) - 1, max_frames, dtype=int)
            self.data = [existing[i] for i in sorted(set(idx))]
            print(f"[{json_path}] subsampled {len(existing)} -> {len(self.data)} frames")
        else:
            self.data = existing

        self.hazy_dir = hazy_dir
        self.gt_dir = gt_dir

    def __len__(self):
        return max(0, len(self.data) - 5)

    # def _load(self, item, folder, ext):
    #     path = os.path.join(folder, os.path.splitext(item['frame'])[0] + ext)
    #     img = cv2.imread(path)
    #     if img is None:
    #         raise FileNotFoundError(f"Could not load {path}")
    #     return img

    def _load(self, item, folder, ext, scale=0.5):
        path = os.path.join(folder, os.path.splitext(item['frame'])[0] + ext)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load {path}")
        
        # Resize image to save GPU memory
        if scale != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img    

    def __getitem__(self, idx):
        max_idx = len(self.data) - 5
        safe_idx = min(idx, max_idx)
        z_item = self.data[safe_idx]
        x_item = self.data[safe_idx + np.random.randint(1, 5)]

        SCALE = 0.25 # Change to 0.25 if you STILL run out of memory

        bx, by, bw, bh = x_item['bbox']
        if bw <= 0 or bh <= 0:
            x_item = z_item

        # z_hazy = self._load(z_item, self.hazy_dir, '.png')
        # x_hazy = self._load(x_item, self.hazy_dir, '.png')
        # z_gt = self._load(z_item, self.gt_dir, '.jpg')
        # x_gt = self._load(x_item, self.gt_dir, '.jpg')

        z_hazy = self._load(z_item, self.hazy_dir, '.png', scale=SCALE)
        x_hazy = self._load(x_item, self.hazy_dir, '.png', scale=SCALE)
        z_gt = self._load(z_item, self.gt_dir, '.jpg', scale=SCALE)
        x_gt = self._load(x_item, self.gt_dir, '.jpg', scale=SCALE)

        # Scale the bounding boxes to match the resized images!
        z_box = np.array(z_item['bbox'], dtype=np.float32) * SCALE
        x_box = np.array(x_item['bbox'], dtype=np.float32) * SCALE        

        # NOTE: assumes all frames in a video share the same resolution, so
        # default_collate can stack these into a batch. True across a single
        # video; don't mix videos of different resolutions in one DataLoader
        # batch without a custom collate_fn.

        # return {
        #     'z_hazy': z_hazy, 'x_hazy': x_hazy,
        #     'z_gt': z_gt, 'x_gt': x_gt,
        #     'z_box': np.array(z_item['bbox'], dtype=np.float32),
        #     'x_box': np.array(x_item['bbox'], dtype=np.float32),
        # }
        return {
            'z_hazy': z_hazy, 'x_hazy': x_hazy,
            'z_gt': z_gt, 'x_gt': x_gt,
            'z_box': z_box,
            'x_box': x_box,
        }

def build_datasets(json_dir_triples, max_frames=None):
    """json_dir_triples: list of (json_path, hazy_dir, gt_dir) tuples."""
    datasets = [JointDataset(j, h, g, max_frames) for j, h, g in json_dir_triples]
    return ConcatDataset(datasets)


# ==============================================================================
# Differentiable dehazing (mirrors unified_wrapper.dehaze, but stays as a
# float tensor with gradients enabled instead of converting to uint8 numpy)
# ==============================================================================
def dehaze_batch(dehazer, hazy_bgr_uint8_batch, device):
    """
    hazy_bgr_uint8_batch: [B, H, W, C] uint8 tensor (BGR, from cv2 via default_collate)
    Returns: [B, 3, H, W] float tensor, BGR, 0-255 range (matches the scale/
    order the tracker was originally fed in cotrain_finetune_tracker.py -
    raw cv2 images with NO normalization), gradients enabled.
    """
    imgs = hazy_bgr_uint8_batch.to(device).float() / 255.0   # [B,H,W,C] BGR 0-1
    imgs = imgs.flip(-1)                                      # BGR -> RGB
    imgs = imgs.permute(0, 3, 1, 2).contiguous()               # [B,3,H,W] RGB 0-1

    factor = 32
    _, _, h, w = imgs.shape
    H = ((h + factor - 1) // factor) * factor
    W = ((w + factor - 1) // factor) * factor
    padh, padw = H - h, W - w
    imgs_padded = F.pad(imgs, (0, padw, 0, padh), mode='reflect')

    clean_list = dehazer(imgs_padded)
    clean = clean_list[2][:, :, :h, :w]   # RGB, 0-1, differentiable, [B,3,h,w]

    clean_bgr_255 = clean.flip(1) * 255.0  # RGB->BGR (flip channel dim), scale to 0-255
    return clean_bgr_255


def load_gt_batch(gt_bgr_uint8_batch, device):
    """Same target format as dehaze_batch's output, but from real GT images (no grad needed)."""
    imgs = gt_bgr_uint8_batch.to(device).float()          # [B,H,W,C] BGR 0-255
    imgs = imgs.permute(0, 3, 1, 2).contiguous()            # [B,3,H,W] BGR 0-255
    return imgs


# ==============================================================================
# Differentiable crop (mirrors SiameseTrackerDataset._get_crop from
# cotrain_finetune_tracker.py, but operates on torch tensors so gradients
# flow back through the crop into the dehazer)
# ==============================================================================
def compute_crop_geometry(box, jitter=False, center_jitter=0.3, scale_jitter=0.1, out_size=EXEMPLAR_SIZE):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    context = CONTEXT_AMOUNT * (w + h)
    s_z = float(np.sqrt(max(1e-6, (w + context) * (h + context))))

    # Scale the crop side to the canvas we're resizing into. s_z (above) is
    # the template/exemplar-scale crop side. The search crop must capture a
    # proportionally LARGER region of the original image so that, after
    # resizing to out_size, the target keeps the same ABSOLUTE pixel size it
    # has in the template - which is the size the anchors are calibrated for.
    # This mirrors PySOT's own crop_like_SiamFC: s_x = s_z * (instance_size /
    # exemplar_size). Without this, out_size=SEARCH_SIZE (255) reuses the
    # exemplar-sized region (127-scale) stretched to fill a 255 canvas, so
    # the target ends up ~2x larger than any anchor -> IoU never clears
    # cfg.TRAIN.THR_HIGH -> zero positive anchors -> loc_loss stuck at 0.
    s = s_z * (out_size / EXEMPLAR_SIZE)

    if jitter:
        if center_jitter > 0:
            cx += np.random.uniform(-center_jitter, center_jitter) * s
            cy += np.random.uniform(-center_jitter, center_jitter) * s
        if scale_jitter > 0:
            s *= np.random.uniform(1 - scale_jitter, 1 + scale_jitter)

    return {'cx': cx, 'cy': cy, 's': s}


def extract_torch_crop(img_1chw, geom, out_size):
    """img_1chw: [1, 3, H, W] float tensor, differentiable."""
    _, C, H, W = img_1chw.shape
    cx, cy, s = geom['cx'], geom['cy'], geom['s']
    half = s / 2.0
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half

    pad_left = int(max(0, -x1))
    pad_top = int(max(0, -y1))
    pad_right = int(max(0, x2 - W))
    pad_bottom = int(max(0, y2 - H))

    if pad_left or pad_top or pad_right or pad_bottom:
        img_padded = F.pad(img_1chw, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
    else:
        img_padded = img_1chw

    x1p = max(0, int(round(x1 + pad_left)))
    y1p = max(0, int(round(y1 + pad_top)))
    x2p = min(img_padded.shape[3], int(round(x1p + s)))
    y2p = min(img_padded.shape[2], int(round(y1p + s)))

    window = img_padded[:, :, y1p:y2p, x1p:x2p]
    if window.shape[2] < 2 or window.shape[3] < 2:
        window = img_padded  # degenerate fallback

    crop = F.interpolate(window, size=(out_size, out_size), mode='bilinear', align_corners=False)
    return crop  # [1, 3, out_size, out_size]

def box_to_crop_coords(box, geom, out_size):
    x, y, w, h = box
    cx, cy, s = geom['cx'], geom['cy'], geom['s']
    scale = out_size / s
    origin_x, origin_y = cx - s / 2.0, cy - s / 2.0
    
    # 1. Map top-left corner to the crop's pixel space
    x_mapped = (x - origin_x) * scale
    y_mapped = (y - origin_y) * scale
    
    # 2. STRICT CLAMPING: Width and Height must be >= 1.0 pixel 
    # This specifically prevents the log(0) NaN crash
    w_mapped = max(1.0, w * scale)
    h_mapped = max(1.0, h * scale)
    
    # 3. Convert to CORNER coordinates [x1, y1, x2, y2].
    # PySOT's AnchorTarget.__call__ internally does
    # `tcx, tcy, tw, th = corner2center(target)`, i.e. it expects a corner-form
    # box, NOT center-form. Passing [cx, cy, w, h] here (as the old code did)
    # gets silently misinterpreted as [x1, y1, x2, y2], so corner2center
    # computes tw = w_mapped - center_x, which is negative whenever the box is
    # smaller than the crop center coordinate (i.e. almost always) -> feeds a
    # negative width/height into np.log() downstream -> NaN loc loss.
    x1 = x_mapped
    y1 = y_mapped
    x2 = x_mapped + w_mapped
    y2 = y_mapped + h_mapped

    return [x1, y1, x2, y2]

# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Joint end-to-end co-training of dehazer + tracker RPN head")
    parser.add_argument('--train_manifests', type=str, nargs='+', required=True)
    parser.add_argument('--hazy_dirs', type=str, nargs='+', required=True,
                        help='Raw HAZY frame directories, SAME ORDER as --train_manifests')
    parser.add_argument('--gt_dirs', type=str, nargs='+', required=True,
                        help='Clean GT frame directories (.jpg), SAME ORDER as --train_manifests')
    parser.add_argument('--dehaze_ckpt', type=str, default='checkpoints/ots.pkl')
    parser.add_argument('--track_ckpt', type=str, default='checkpoints/siamrpn_r50.pth')
    parser.add_argument('--track_config', type=str, default='pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
    parser.add_argument('--out_prefix', type=str, default='checkpoints/joint_v1')
    parser.add_argument('--lr_dehaze', type=float, default=1e-4)
    parser.add_argument('--lr_tracker', type=float, default=1e-3)
    parser.add_argument('--pixel_loss_weight', type=float, default=1.0,
                        help='Weight on the dehazer-output-vs-GT L1 pixel loss. This is the '
                             'anchor that keeps the dehazer from drifting into pixels that only '
                             'fool the tracker without being genuinely useful/realistic.')
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Lower than the decoupled script default (8) - each sample now costs '
                             '2 full-frame dehazer forward passes, more memory-hungry.')
    parser.add_argument('--center_jitter', type=float, default=0.3)
    parser.add_argument('--scale_jitter', type=float, default=0.1)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (len(args.train_manifests) == len(args.hazy_dirs) == len(args.gt_dirs)):
        raise ValueError("--train_manifests, --hazy_dirs, --gt_dirs must all have matching length/order.")

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    cfg.merge_from_file(args.track_config)

    # --- Dehazer: TRAINABLE this time ---
    dehazer = build_eenet().to(device)
    dehaze_state = torch.load(args.dehaze_ckpt, map_location=device, weights_only=True)
    dehazer.load_state_dict(dehaze_state['model'])
    dehazer.train()

    # --- Tracker: backbone frozen, RPN head trainable (same as before) ---
    track_model = ModelBuilder().to(device)
    track_model.load_state_dict(torch.load(args.track_ckpt, map_location=device))
    for param in track_model.backbone.parameters():
        param.requires_grad = False
    track_model.rpn_head.train()

    anchor_target = AnchorTarget()

    optimizer = torch.optim.Adam([
        {'params': dehazer.parameters(), 'lr': args.lr_dehaze},
        {'params': track_model.rpn_head.parameters(), 'lr': args.lr_tracker},
    ])

    triples = list(zip(args.train_manifests, args.hazy_dirs, args.gt_dirs))
    dataset = build_datasets(triples, args.max_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Joint training on GPU {args.gpu} | {len(triples)} video(s) | "
          f"lr_dehaze={args.lr_dehaze} lr_tracker={args.lr_tracker} "
          f"pixel_weight={args.pixel_loss_weight} batch_size={args.batch_size}")

    # Initialize the AMP Scaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            B = batch['z_hazy'].shape[0]

            # Clear gradients at the start of the loop
            optimizer.zero_grad()

            # 1. Dehaze template and search frames (Standard FP32 - Prevents cuFFT crash)
            z_clean = dehaze_batch(dehazer, batch['z_hazy'], device)   
            x_clean = dehaze_batch(dehazer, batch['x_hazy'], device)   
            z_gt_full = load_gt_batch(batch['z_gt'], device)
            x_gt_full = load_gt_batch(batch['x_gt'], device)

            template_crops, search_crops = [], []
            cls_list, delta_list, weight_list = [], [], []
            pixel_losses = []

            # 2. Per-sample crop (geometry differs per box)
            for b in range(B):
                z_box = batch['z_box'][b].numpy()
                x_box = batch['x_box'][b].numpy()

                z_geom = compute_crop_geometry(z_box, jitter=False, out_size=EXEMPLAR_SIZE)  # template: no jitter
                x_geom = compute_crop_geometry(x_box, jitter=True,
                                                center_jitter=args.center_jitter,
                                                scale_jitter=args.scale_jitter,
                                                out_size=SEARCH_SIZE)

                z_crop = extract_torch_crop(z_clean[b:b+1], z_geom, EXEMPLAR_SIZE)
                x_crop = extract_torch_crop(x_clean[b:b+1], x_geom, SEARCH_SIZE)
                template_crops.append(z_crop)
                search_crops.append(x_crop)

                # Pixel loss: dehazer's crop vs REAL GT crop, same geometry
                z_gt_crop = extract_torch_crop(z_gt_full[b:b+1], z_geom, EXEMPLAR_SIZE)
                x_gt_crop = extract_torch_crop(x_gt_full[b:b+1], x_geom, SEARCH_SIZE)
                pixel_losses.append(F.l1_loss(z_crop, z_gt_crop) + F.l1_loss(x_crop, x_gt_crop))

                pysot_bbox = box_to_crop_coords(x_box, x_geom, SEARCH_SIZE)
                cls, delta, delta_weight, _ = anchor_target(pysot_bbox, cfg.TRAIN.OUTPUT_SIZE)
                cls_list.append(torch.from_numpy(cls))
                delta_list.append(torch.from_numpy(delta))
                weight_list.append(torch.from_numpy(delta_weight))

            template_batch = torch.cat(template_crops, dim=0)   # [B,3,127,127]
            search_batch = torch.cat(search_crops, dim=0)       # [B,3,255,255]
            pixel_loss = torch.stack(pixel_losses).mean()

            track_batch = {
                'template': template_batch,
                'search': search_batch,
                'label_cls': torch.stack(cls_list).to(device),
                'label_loc': torch.stack(delta_list).to(device),
                'label_loc_weight': torch.stack(weight_list).to(device),
            }

            # 3. Tracker in FP16 (Autocast limits memory spikes)
            with torch.cuda.amp.autocast():
                outputs = track_model(track_batch)
                cls_loss = outputs['cls_loss']
                loc_loss = outputs['loc_loss']
                track_loss = cls_loss + loc_loss
                total_loss = track_loss + args.pixel_loss_weight * pixel_loss

            # 4. Scale the loss and backpropagate
            scaler.scale(total_loss).backward()
            
            # Unscale before clipping gradients
            if args.clip_norm > 0:
                scaler.unscale_(optimizer)
                # Clip each parameter group SEPARATELY. pixel_loss (dehazer-
                # only gradient source) runs ~10-50x larger than track_loss,
                # so a single combined clip_grad_norm_ over both groups lets
                # the dehazer's large pixel-driven gradients dominate the
                # global norm and drag the RPN head's much smaller gradients
                # down with it, even when the RPN head's own gradients never
                # needed clipping. Clipping independently keeps the tracker's
                # learning signal from being suppressed by the dehazer's.
                torch.nn.utils.clip_grad_norm_(dehazer.parameters(), max_norm=args.clip_norm)
                torch.nn.utils.clip_grad_norm_(track_model.rpn_head.parameters(), max_norm=args.clip_norm)
            
            # Step the optimizer and update the scale for next iteration
            scaler.step(optimizer)
            scaler.update()

            if i % args.log_every == 0:
                weighted_pixel = args.pixel_loss_weight * pixel_loss.item()
                print(f"Epoch {epoch} | Iter {i} | cls={cls_loss.item():.4f} loc={loc_loss.item():.4f} "
                      f"track={track_loss.item():.4f} pixel_raw={pixel_loss.item():.4f} "
                      f"pixel_weighted={weighted_pixel:.4f} total={total_loss.item():.4f}")

        torch.save({'model': dehazer.state_dict()}, f"{args.out_prefix}_dehazer_epoch{epoch}.pkl")
        torch.save(track_model.state_dict(), f"{args.out_prefix}_tracker_epoch{epoch}.pth")
        print(f"Saved epoch {epoch} checkpoints.")

    print("Joint training complete. Validate EACH epoch's (dehazer, tracker) pair together "
          "with eval_parallel.py + compare_tracking_accuracy.py on a held-out video.")
    
    
if __name__ == "__main__":
    main()