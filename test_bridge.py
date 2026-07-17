import cv2
import torch
from unified_wrapper import UnifiedDehazeTracker

from paths import DEHAZE_CKPT, TRACK_MODEL, TRACK_CONFIG, VIDEO_PATH, OUTPUT_DIR
DEHAZE_MODEL = "checkpoints/eenet_ots.pkl" 
TRACK_MODEL = "checkpoints/siamrpn_r50.pth"
TRACK_CONFIG = "pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml"

# 2. Initialize the Bridge
print("Initializing models... this may take a moment.")
agent = UnifiedDehazeTracker(DEHAZE_MODEL, TRACK_MODEL, TRACK_CONFIG)

# 3. Load a test image from your data
# Pick any hazy image from your /data folder
test_img_path = "../IIT_HAZY/vid3/hazy/00000001.png" 
frame = cv2.imread(test_img_path)

if frame is None:
    print(f"Error: Could not find image at {test_img_path}")
else:
    # 4. Test Dehazing
    clean_frame = agent.dehaze(frame)
    cv2.imwrite("test_dehazed_output.jpg", clean_frame)
    print("Dehazing successful! Check 'test_dehazed_output.jpg'")

    # 5. Test Tracking Init
    # Hardcode a box for the test: [x, y, width, height]
    test_box = [576, 547, 912, 480]
    agent.init_tracker(clean_frame, test_box)
    print("Tracker initialization successful!")