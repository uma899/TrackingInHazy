Download weights https://drive.google.com/drive/folders/1vgdunAsTCHZECrChV_c98yMUr8VTz4pI?usp=sharing
and put those two folders in root folder

### TESTS - Total 10

test1 - model_80, siam..
test2 - As object is losing track after trees, search area is wideened.
test2_2 - Improving zooom conditions
test4 - finetune siam.. with direct hazy images

test5 - with vid8 for training 

    TEST 5 results:
    Use results.py file to compare json files of gt and detracked ones. You get output something like - 
                                Mean IoU:                      0.4803
                                Success Rate @ IoU > 0.50:      0.6079
                                Success Plot AUC (0-1):        0.4804
                                Mean Center Error (px):        64.24
                                Precision @ 20px:            0.7562
                                Precision Plot AUC (0-50px):   0.6876
                                Full Failure Rate (IoU <= 0.00): 0.0972

    Run this snippet to find which part of video tracker is confusing:
    import csv
    with open('per_frame_results_vid3.csv') as f:
        rows = [r for r in csv.DictReader(f) if float(r['iou']) <= 0.0]
        for r in rows:
            print(r['frame'], r['iou'])                            


    TEST 5 includes
        Before fine tune: Output: out_direct_track_test5.mp4
        In test 5, "vid8" is used to train and get finetuned model files for Dehazer (eenet_test5.pkl) and Tracker (siamrpn_test5.pth).
        Then after training, its is directly tested on on "vid3" and found output tracking is not lost after drone crosses trees. 
        Output (dehaze then track using new models): train_vid8_test_vid3.mp4, vid8_test5
        Output (direct track with new models): out_direct_track_test5_vid3.mp4  -> Confused after trees, and retracked after





test6 - code changes (jitter)
Until above, its sequential and its finetuning. From below 
joint startts... with joint_train_dehaze_tracker.py

test7 - joint_v1.pth - train and test on vid8 - NO IMPROVEMENT
test8 - on vid3 - Done bad. Lost track
test9 - Training for more epochs (6) on lesser data size (600 frames). Found it sharply lost 
track as tree background comes.

test10 - unified_wrapper changes. whenever confidence is above conf_threshold (default 0.97). When a frame's score drops below that, instead of letting PySOT commit to that frame's argmax location, it rolls self.tracker.center_pos/self.tracker.size back to the last confident state. And now tracker worked, though confused at trees. vid3_joint_shorted_IMP/ contains 
cotrained weights

test11 - Training with full data (vid3). Testing on vid8. Full data trained weights are bad compared to 
shorted data


### BOX ccordinates:
vid3: 698 499 248 102
vid8: 604 624 908 422


----

### NOTE

model_weights folder contain original model weights

----

### FILES

## a. Co-Training

joint....py

## b. Fine-tuning => Found uselesss

## 1.Supervised
genera_gt.. - To get bounding box coordinates from gt
supervised_.._dehazer - To get eenet_supervised.pkl
restore_frames_supervised.py - To get dehazed images using new weights
cot.._fine.._tracker - To get siamrpn_restored_finetuned.pth
compare_models - 

pre.._pro.._re.._data - Creates dehazed images in restored_supervised folder


## 2. Unsupervised  - used when gt not available
cotrain_infer... - Draw boxes based on dehazing hazy ones and save coordinates as json
cot.._fine.._dehazer - To get eenet_cotrained.pkl
cot.._fine.._tracker (same file as above) - To get siamrpn_cotrained.pth
compare_models - 



---------
