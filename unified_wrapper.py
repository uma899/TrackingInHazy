import sys
import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F  # Make sure this is at the top

# Force deterministic behavior. Without this, cuDNN's algorithm
# auto-tuning can select different (numerically slightly different)
# convolution implementations depending on which process/context loaded
# the model, even with identical weights and identical GPU hardware. Those
# tiny differences compound across a deep network and can flip a fragile
# tracker's prediction - which is what was happening between
# eval_parallel.py's subprocess-loaded dehazer and debug_track_raw.py's
# in-process one.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Add paths to system so python can see the modules
sys.path.append(os.path.join(os.getcwd(), 'EENet/OTS'))
sys.path.append(os.path.join(os.getcwd(), 'pysot'))

from models.EENet import build_net as build_eenet
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.core.config import cfg

class UnifiedDehazeTracker:
    # Must match SCALE in joint_train_dehaze_tracker.py. Both the dehazer
    # and the tracker were trained exclusively on frames/boxes downscaled by
    # this factor - the dehazer never saw full-resolution input, and the
    # tracker's box-size calibration (search-region scale, anchor matching)
    # is tied to boxes at this scale. Running eval at native resolution feeds
    # both networks out-of-distribution inputs ~1/SCALE times larger than
    # anything seen in training, which is why the box balloons immediately.
    SCALE = 0.25

    def __init__(self, dehaze_path, track_path, track_config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Load Dehazer (EENet)
        self.dehazer = build_eenet()
        state = torch.load(dehaze_path, map_location=self.device)
        self.dehazer.load_state_dict(state['model'])
        self.dehazer.to(self.device).eval()

        # 2. Load Tracker (PySOT)
        cfg.merge_from_file(track_config)
        self.track_model = ModelBuilder()
        self.track_model.load_state_dict(torch.load(track_path, map_location=self.device))
        self.track_model.to(self.device).eval()
        self.tracker = build_tracker(self.track_model)



# ... (inside UnifiedDehazeTracker class)

    def dehaze(self, frame):
        """ Returns a dehazed BGR image (at SCALE resolution, matching
        training) with automatic padding to multiple of 32 """
        # 0. Downscale to match training resolution - the dehazer was never
        # trained on full-resolution input.
        if self.SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=self.SCALE, fy=self.SCALE,
                                interpolation=cv2.INTER_AREA)

        # 1. Convert BGR to RGB and Tensor
        img = frame[:, :, ::-1].transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).float().unsqueeze(0).to(self.device) / 255.0
        
        # 2. Handle Padding (Factor of 32)
        factor = 32
        _, _, h, w = img.shape
        H, W = ((h + factor - 1) // factor) * factor, ((w + factor - 1) // factor) * factor
        padh = H - h
        padw = W - w
        
        # Pad only the right and bottom
        img_padded = F.pad(img, (0, padw, 0, padh), mode='reflect')

        with torch.no_grad():
            # Run the model on the padded image
            # EENet returns a list; index 2 is the final high-res output
            clean_tensor_list = self.dehazer(img_padded)
            clean_tensor = clean_tensor_list[2]
            
            # 3. Crop back to original dimensions
            clean_tensor = clean_tensor[:, :, :h, :w]
            
        # 4. Post-processing to CV2 image
        clean_img = clean_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        clean_img = np.clip(clean_img * 255, 0, 255).astype(np.uint8)
        
        return clean_img[:, :, ::-1].copy() # Return BGR
    def init_tracker(self, frame, bbox):
        # frame here is the already-dehazed SCALE-resolution image (written
        # by dehaze() during Phase 1). bbox is expected in ORIGINAL
        # resolution (e.g. straight from cv2.selectROI / gt_labels.json) -
        # scale it down to match, since the tracker's internal search/anchor
        # geometry was calibrated on SCALE-resolution boxes during training.
        scaled_bbox = [c * self.SCALE for c in bbox]
        self.tracker.init(frame, scaled_bbox)
        # Track our own "last confident state" separately from whatever
        # PySOT's tracker.center_pos/size get overwritten to internally -
        # this is what confidence-gating rolls back to.
        self._last_confident_center = self.tracker.center_pos.copy()
        self._last_confident_size = self.tracker.size.copy()

    def track_frame(self, frame, conf_threshold=0.8):
        # PySOT's tracker.track() unconditionally commits to its argmax
        # response location every frame - it never checks whether that
        # response is actually trustworthy before moving the search window
        # there. One low-confidence frame (e.g. a distractor in tree
        # clutter) permanently drags the search window away from the real
        # object, since every subsequent frame's search region is centered
        # on wherever the PREVIOUS frame ended up. Gate on score: if this
        # frame's confidence is too low to trust, roll the tracker's
        # internal state back to the last confident position/size instead
        # of letting it commit to the drift, so the search window stays
        # anchored near the real object and has a chance to reacquire once
        # conditions improve, rather than compounding the drift forever.
        res = self.tracker.track(frame)

        if res['best_score'] < conf_threshold:
            self.tracker.center_pos = self._last_confident_center.copy()
            self.tracker.size = self._last_confident_size.copy()

            cx, cy = self._last_confident_center
            w, h = self._last_confident_size
            res = dict(res)
            res['bbox'] = [float(cx - w / 2.0), float(cy - h / 2.0), float(w), float(h)]
        else:
            self._last_confident_center = self.tracker.center_pos.copy()
            self._last_confident_size = self.tracker.size.copy()

        # Convert the predicted bbox back to ORIGINAL resolution so callers
        # (eval_parallel.py's JSON output, results.py comparison against
        # gt_json) work in the same coordinate space the ground truth is in.
        res = dict(res)
        res['bbox'] = [c / self.SCALE for c in res['bbox']]
        return res
    
    
# import sys
# import os
# import torch
# import numpy as np
# import cv2
# import torch.nn.functional as F  # Make sure this is at the top

# # Force deterministic behavior. Without this, cuDNN's algorithm
# # auto-tuning can select different (numerically slightly different)
# # convolution implementations depending on which process/context loaded
# # the model, even with identical weights and identical GPU hardware. Those
# # tiny differences compound across a deep network and can flip a fragile
# # tracker's prediction - which is what was happening between
# # eval_parallel.py's subprocess-loaded dehazer and debug_track_raw.py's
# # in-process one.
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# # Add paths to system so python can see the modules
# sys.path.append(os.path.join(os.getcwd(), 'EENet/OTS'))
# sys.path.append(os.path.join(os.getcwd(), 'pysot'))

# from models.EENet import build_net as build_eenet
# from pysot.models.model_builder import ModelBuilder
# from pysot.tracker.tracker_builder import build_tracker
# from pysot.core.config import cfg

# class UnifiedDehazeTracker:
#     # Must match SCALE in joint_train_dehaze_tracker.py. Both the dehazer
#     # and the tracker were trained exclusively on frames/boxes downscaled by
#     # this factor - the dehazer never saw full-resolution input, and the
#     # tracker's box-size calibration (search-region scale, anchor matching)
#     # is tied to boxes at this scale. Running eval at native resolution feeds
#     # both networks out-of-distribution inputs ~1/SCALE times larger than
#     # anything seen in training, which is why the box balloons immediately.
#     SCALE = 0.25

#     def __init__(self, dehaze_path, track_path, track_config):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # 1. Load Dehazer (EENet)
#         self.dehazer = build_eenet()
#         state = torch.load(dehaze_path, map_location=self.device)
#         self.dehazer.load_state_dict(state['model'])
#         self.dehazer.to(self.device).eval()

#         # 2. Load Tracker (PySOT)
#         cfg.merge_from_file(track_config)
#         self.track_model = ModelBuilder()
#         self.track_model.load_state_dict(torch.load(track_path, map_location=self.device))
#         self.track_model.to(self.device).eval()
#         self.tracker = build_tracker(self.track_model)



# # ... (inside UnifiedDehazeTracker class)

#     def dehaze(self, frame):
#         """ Returns a dehazed BGR image (at SCALE resolution, matching
#         training) with automatic padding to multiple of 32 """
#         # 0. Downscale to match training resolution - the dehazer was never
#         # trained on full-resolution input.
#         if self.SCALE != 1.0:
#             frame = cv2.resize(frame, None, fx=self.SCALE, fy=self.SCALE,
#                                 interpolation=cv2.INTER_AREA)

#         # 1. Convert BGR to RGB and Tensor
#         img = frame[:, :, ::-1].transpose(2, 0, 1).copy()
#         img = torch.from_numpy(img).float().unsqueeze(0).to(self.device) / 255.0
        
#         # 2. Handle Padding (Factor of 32)
#         factor = 32
#         _, _, h, w = img.shape
#         H, W = ((h + factor - 1) // factor) * factor, ((w + factor - 1) // factor) * factor
#         padh = H - h
#         padw = W - w
        
#         # Pad only the right and bottom
#         img_padded = F.pad(img, (0, padw, 0, padh), mode='reflect')

#         with torch.no_grad():
#             # Run the model on the padded image
#             # EENet returns a list; index 2 is the final high-res output
#             clean_tensor_list = self.dehazer(img_padded)
#             clean_tensor = clean_tensor_list[2]
            
#             # 3. Crop back to original dimensions
#             clean_tensor = clean_tensor[:, :, :h, :w]
            
#         # 4. Post-processing to CV2 image
#         clean_img = clean_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
#         clean_img = np.clip(clean_img * 255, 0, 255).astype(np.uint8)
        
#         return clean_img[:, :, ::-1].copy() # Return BGR
#     def init_tracker(self, frame, bbox):
#         # frame here is the already-dehazed SCALE-resolution image (written
#         # by dehaze() during Phase 1). bbox is expected in ORIGINAL
#         # resolution (e.g. straight from cv2.selectROI / gt_labels.json) -
#         # scale it down to match, since the tracker's internal search/anchor
#         # geometry was calibrated on SCALE-resolution boxes during training.
#         scaled_bbox = [c * self.SCALE for c in bbox]
#         self.tracker.init(frame, scaled_bbox)

#     def track_frame(self, frame):
#         res = self.tracker.track(frame)
#         # Convert the predicted bbox back to ORIGINAL resolution so callers
#         # (eval_parallel.py's JSON output, results.py comparison against
#         # gt_json) work in the same coordinate space the ground truth is in.
#         res = dict(res)
#         res['bbox'] = [c / self.SCALE for c in res['bbox']]
#         return res