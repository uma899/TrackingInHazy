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




But your second question is the **core of your research**: *"How can co-training help if it's the same models?"*

Let me explain the "Co-training Magic" and then give you the code to fix the blur.

---

### 1. Why Co-training is better than Sequential (The "Pilot & Window Cleaner" Analogy)

Imagine a Pilot (**Tracker**) looking through a foggy window that a Window Cleaner (**Dehazer**) is trying to wipe.

*   **Sequential (Standard):** The Window Cleaner wipes the whole glass equally. He doesn't know the Pilot is trying to see a specific landing strip. He might leave a smudge right where the landing strip is, and the Pilot crashes.
*   **Co-training (Your Project):** 
    1.  **Tracker $\rightarrow$ Dehazer:** The Pilot points to a spot and says, *"Clean THIS spot perfectly!"* The Cleaner focuses all his energy there. Even if the rest of the window is messy, that spot is clear.
    2.  **Dehazer $\rightarrow$ Tracker:** The Pilot learns, *"The Cleaner always leaves a slightly blue streak, but I can still see the landing strip's shape through it."* The Pilot adapts to the Cleaner's specific style.

**In technical terms:** You are performing **Domain Specialization**. The original EENet was trained on generic datasets (OTS/ITS). By fine-tuning, you are making it a "Specialist" for the specific lighting, fog density, and objects in **your specific video**.

---

### 3. The "Secret Ingredient" for the Tracker

You asked how tracking gets better. The reason the tracker "fails at a particular instant" in the sequential pipeline is that the **Template** (from Frame 1) and the **Search Image** (Current Frame) look too different.

**The Solution:** In your co-training, you should initialize the tracker on the **Dehazed Frame 1**, not the hazy one. 

**Why this works:**
1.  Your model dehazes Frame 1 (Template).
2.  Your model dehazes Frame 100 (Search).
3.  Because **both** images were processed by the **same** model, they share the **same artifacts**. 
4.  PySOT is great at matching artifacts! It will see the "streaks" or "blur" in both and say, *"Aha! This matches the template I have."*

---

### Next Steps to get that "Better than Sequential" result:

1.  **Modify the loss** (add the `edge_loss` code above) and run the fine-tuning again for 2 epochs. The blur will decrease significantly.
2.  **Fine-tune the Tracker:** Use the `cotrain_finetune_tracker.py` script I gave you earlier. This is the part where the "Pilot" learns to see through the "Cleaner's" work. 
3.  **Final Test:** Use the **new Dehazer** AND the **new Tracker** together. 

**This combination is what creates the "Co-training" state that outperforms simple sequential dehazing.** Give the edge loss a try!
