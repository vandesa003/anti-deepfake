# Anti-deepfake

## Preprocess

- Usage

1. Generate frames, will save the frames under `../dataset/frames/`

   ```bash
   cd ./tools/
   python generate_frames.py path/to/videos --cpu_num 48
   ```

2. Generate face patches, will save the patches under `../dataset/patches/` and will save a csv file `patch_image_statics.csv`.

   ```bash
   cd ./tools/
   python generate_patches.py path/to/frames/
   ```

   `patch_image_statics.csv` contains `PatchName` and `Score`:

   | PatchName                 | Score              |
   | ------------------------- | ------------------ |
   | uaspniazcl_002_face_0.jpg | 0.9996267557144165 |
   | uaspniazcl_003_face_0.jpg | 0.9994097948074341 |
   | uaspniazcl_001_face_0.jpg | 0.9971966743469238 |

   Remember to collect this file after a batch of computation.
   
3. Generate training label csv.

   ```bash
   cd ./tools/
   python anno_parse.py path/to/patch/file path/to/jsonfile --saving_path path/to/save
   ```

- Name rules

1. Frame images:

   `VideoName_FrameId.jpg` eg. `rmufsuogzn_006.jpg`,  `vxawghqzyf_007.jpg`.

2. Patch images:

   `VideoName_FrameId_face_FaceId.jpg` eg. `ucthmsajay_008_face_0.jpg`, `zzlsynxeff_008_face_0.jpg`

## Training

Train the Exception Net. Need to change some path in the code.

```bash
cd ./tools/
python train_net.py
```

After training, the best model and checkpoints will saved under `./saved_models/model.pth`

## Inference

Do inference on video level.

```bash
cd ./tools/
python infer_net.py
```

After inference, the result will be saved under `./dataset/submission.csv`

## TODO List

- [x] XceptionNet
- [x] Basic training logic
- [x] Weighted training
- [x] Inference
- [ ] Data augmentation
- [ ] Better face detection model/ finetune face detection model