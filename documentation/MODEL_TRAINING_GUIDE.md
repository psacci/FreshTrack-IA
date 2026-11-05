# Model training guide

Two models are involved:

| Model | Source | Needs your training? |
|---|---|---|
| YOLOv8n food detector | Pretrained on COCO (downloads automatically), optionally fine-tuned on VegQual or your own labels | No. Custom training is optional. |
| MobileNetV2 freshness classifier | Transfer learning on the fruit dataset, optionally expanded with Freshness44 and/or Fresh and Stale produce images | A baseline checkpoint ships with the project. Retrain for more freshness-supported foods. |

## 1. Freshness classifier

### The dataset

`training/prepare_dataset.py` uses the public **Fruit-freshness-detection-dataset** (github.com/zijianchen98, Apache-2.0): 3,202 photos with 7,087 labelled fruits (apple, banana, orange), each tagged *fresh*, *normal* or *rotten*.

Label mapping used by the app:

| Dataset label | FreshTrack class |
|---|---|
| fresh | Fresh |
| normal | Aging / Overripe |
| rotten | Visible spoilage |

The "normal" to "Aging" mapping is our interpretation of the dataset's middle level of decay. State it as such in your paper.

### Build it (about 5 minutes)

```bash
python training/prepare_dataset.py
python training/check_freshness_dataset.py
```

`prepare_dataset.py` downloads the dataset with `git` (about 350 MB, no Kaggle account), crops each fruit out of its photo, and splits train/val/test **without leakage**:

- The dataset contains many near-identical photos of the same fruit. A random split would put copies in both train and test and inflate the score.
- The script therefore groups every crop that comes from the same photo or looks near-identical (256-bit difference hash, mirrored copies included) and keeps each group inside one split.
- Class proportions stay balanced across splits.

Result: about 4,800 train, 1,060 validation and 1,250 test crops. The folders appear in `dataset/freshness/{train,val,test}/{fresh,aging,spoiled}/` and a `manifest.csv` records each crop's fruit and source photo.

### Train

```bash
python training/train_freshness.py
```

What it does:

1. **Warm-up** (3 epochs): ImageNet-pretrained MobileNetV2 with the backbone frozen; only the new final layer learns.
2. **Fine-tune** (up to 12 epochs): the whole network learns at a low learning rate with cosine decay and early stopping.
3. Rare classes get a larger weight in the loss. Augmentation: random crop, flip, small rotation, brightness/contrast. No hue shift, because colour carries the freshness signal.
4. The checkpoint with the best **validation macro-F1** is saved to `models/freshness_classification/mobilenetv2_freshness.pt`. Training curves go to `training_curves.png`.

Useful options:

```bash
python training/train_freshness.py --epochs 8 --batch 16      # lighter on memory
python training/train_freshness.py --image-size 160           # faster, slightly less detail
python training/train_freshness.py --workers 0                # if data loading errors on macOS
```

Apple Silicon uses the GPU (MPS) automatically, falling back to CPU. I could not test MPS here. As a rough guide, the 224 px run should take tens of minutes on a recent Mac; if it is slow, use `--image-size 160`.

### Evaluate (this produces the numbers for your paper)

```bash
python training/evaluate_freshness.py
```

Writes to a checkpoint-specific `*_evaluation/` directory by default: `metrics.json`, `classification_report.txt`, `confusion_matrix.png` and `misclassified.csv`. The About page only displays a report whose checkpoint filename matches the loaded checkpoint. The test split is never used for training or for choosing the checkpoint.

### Test on your own photos (strongly recommended)

Clean dataset photos are easier than real life. To measure real-world performance:

1. Photograph about 30 to 60 real items from the foods supported by your checkpoint, in your kitchen and in different light.
2. Sort them into `dataset/own_photos/fresh/`, `aging/` and `spoiled/`.
3. Run:

```bash
python training/evaluate_freshness.py --extra-dir dataset/own_photos
```

You get a separate "real world" accuracy. If it is lower than the test score, say so in the paper; that gap is a finding, not a failure.

### The default checkpoint and other evaluation results

The app's default `models/freshness_classification/mobilenetv2_freshness.pt` checkpoint was trained in a 1-CPU sandbox at **128 px**: 3 head-only epochs plus 2 short fine-tuning passes. Its held-out report covers 1,246 crops:

| Metric | Value |
|---|---|
| Accuracy | 96.9% |
| Macro F1 | 0.970 |
| Fresh (precision / recall) | 0.980 / 0.930 |
| Aging (precision / recall) | 0.952 / 0.993 |
| Spoilage (precision / recall) | 0.992 / 0.976 |

Most errors are *fresh* fruit predicted as *aging* (27 of 417). These are research-dataset results, not household performance.

A separate `mobilenetv2_fresh_stale.pt` checkpoint was evaluated on the expanded six-food test split (3,294 images): accuracy 98.27%, macro-F1 0.965. Spoiled precision/recall was 0.908/0.927 on 245 images. This is not the app's default checkpoint. Bitter gourd, capsicum, and tomato have no spoiled test images, so the report cannot establish spoiled-class performance for those foods. The split contains apples, bananas, bitter gourd, capsicum, oranges, and tomatoes.

**Deadline/provenance warning:** the checkpoint metadata timestamps currently embedded in both freshness checkpoints (November 7 and November 8, 2026) are later than the stated November 6 submission deadline. The expanded evaluation report itself is dated November 6, but that does not resolve the checkpoint timestamp conflict. Verify the provenance/history before describing either model result as work completed before the deadline.

The produce detector's held-out LVIS test report covers 180 images. It reports mAP50 0.1815 and mAP50-95 0.1088; cucumber AP50 is 0.267 and banana AP50 is 0.834. Include these weaker results too; do not imply that adding classes made detection reliable. No household-photo evaluation or custom meat/dairy detector training has been completed.

### Notes for the paper

- Describe the leakage-free split. It is a methodological strength.
- Be explicit that the dataset is photographed in controlled conditions and that the "aging" class maps the dataset's "normal" level.
- The model judges visible appearance only. It cannot detect bacteria or toxins.

### Expanding freshness support with Freshness44

[Freshness44](https://www.kaggle.com/datasets/siavash93/freshness44) contains 53,616 fresh/rotten images for 22 produce types, is about 7.4 GB, and is licensed CC BY-SA 4.0. It is a visual classification dataset, not a food-safety dataset. Download it with your own Kaggle account; do not share or commit your Kaggle credentials. Install/configure the Kaggle CLI locally, then run:

```bash
python -m pip install kaggle
python -m training.prepare_dataset
kaggle datasets download -d siavash93/freshness44 -p data/raw/freshness44 --unzip
python -m training.prepare_freshness44
python -m training.check_freshness_dataset --data dataset/freshness_expanded
python -m training.train_freshness --data dataset/freshness_expanded
python -m training.evaluate_freshness --data dataset/freshness_expanded
```

The preparation script combines Freshness44 with the original apple/banana/orange dataset. It maps Freshness44's fresh and rotten folders to `fresh` and `spoiled`; the original fruit dataset still supplies the `aging` class. Its deterministic image-level split, `manifest.csv`, and `ATTRIBUTION.txt` are written under `dataset/freshness_expanded/`. The trained checkpoint records the food names from that manifest, and the app uses that list to decide when to offer freshness inference. Keep attribution and comply with CC BY-SA 4.0 if redistributing adapted material or a resulting model. The number of foods in the checkpoint is not a measure of model accuracy; separately evaluate on household photos.

### Expanding with Fresh and Stale produce images

Kaggle lists [Fresh and Stale Images of Fruits and Vegetables](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables) as CC0, about 1.5 GB, and covering apples, bananas, bitter gourd, capsicum, oranges, and tomatoes. Download and extract it locally with your own Kaggle account:

```bash
python -m pip install kaggle
kaggle datasets download -d raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables -p data/raw/fresh-stale --unzip
python -m training.prepare_dataset
python -m training.prepare_fresh_stale
python -m training.check_freshness_dataset --data dataset/freshness_stale_expanded
python -m training.train_freshness --data dataset/freshness_stale_expanded
python -m training.evaluate_freshness --data dataset/freshness_stale_expanded
```

The script merges the dataset with the existing three-class set. It maps source `fresh` to `fresh`, and source `stale` to `aging` (not `spoiled`); treat that mapping as an approximation because the source's categories are not identical to the existing dataset's `normal` label. The source author says some images were borrowed from another dataset and some were extracted from videos. The script removes exact duplicates against the base set and keeps images within a six-bit perceptual-hash radius in one split. This helps reduce leakage, but it cannot prove that all related video frames or reused images have been found. Inspect `manifest.csv` and `ATTRIBUTION.txt`; while Kaggle lists CC0, verify upstream rights for reused material before redistribution.

To combine this data with Freshness44 as well, first prepare the Fresh/Stale set as above, then merge Freshness44 into it:

```bash
python -m training.prepare_freshness44 --base dataset/freshness_stale_expanded --out dataset/freshness_all
python -m training.check_freshness_dataset --data dataset/freshness_all
python -m training.train_freshness --data dataset/freshness_all \
  --out models/freshness_classification/fresh44/mobilenetv2_fresh44.pt
python -m training.evaluate_freshness --data dataset/freshness_all \
  --ckpt models/freshness_classification/fresh44/mobilenetv2_fresh44.pt
```

Run the app with the trained produce detector and freshness classifier by setting `FRESHTRACK_YOLO` and `FRESHTRACK_FRESHNESS_CKPT` to those checkpoint paths. Confident freshness results are saved on scanned items and shown in **Fridge → Manage → Freshness info**. Recognition coverage and freshness coverage are separate: only foods listed in the loaded freshness checkpoint get an appearance estimate.

Kaggle downloads require user-managed Kaggle access. Do not share or commit Kaggle credentials or tokens.

## 2. YOLOv8 detector

The live camera uses the configured YOLO detector (by default, the 63-class FreshTrack produce checkpoint when present, otherwise `yolov8n.pt`). Photo scans additionally run `yolov8s-worldv2.pt`, a local open-vocabulary model prompted for hot dogs, meat, seafood, dairy, bakery, and packaged foods. Its approximately 25 MB weights download once if missing into `models/food_detection/`; it is not trained on a FreshTrack meat/dairy dataset. Its suggestions should be confirmed.

The Scan page also has a live browser-camera mode. It runs the configured detector on video frames and uses the trained freshness checkpoint for detected foods listed by that checkpoint. The broader open-vocabulary detector is currently used for photo scans only. Other recognized foods are not automatically freshness-classified. Tesseract OCR also reads visible package text (for example, “milk” or “full cream”); this is text extraction, not a trained milk detector, and can fail when text is small, angled, blurry or obscured. Confirm OCR suggestions before adding them to inventory. The browser must grant camera permission; remote deployments need HTTPS and WebRTC networking configuration. The browser stream runs only while the page is open and camera is started; unattended, always-on fridge monitoring and persistent object tracking are not implemented.

To recognise more foods you need your own labelled photos:

1. Put images in `dataset/detection/images/{train,val}/` and YOLO-format labels in `dataset/detection/labels/{train,val}/`.
2. Copy `dataset/detection/data.yaml.example` to `data.yaml` and list your classes.
3. Train: `python training/train_yolo.py --epochs 60`
4. Use it: set `FRESHTRACK_YOLO=models/food_detection/custom_food.pt` in `.env`; add new class names to `YOLO_TO_CATEGORY` in `config/settings.py` if they are not already mapped.

### Choosing and preparing a dataset

#### VegQual: seven vegetables with visible quality labels

[VegQual](https://figshare.com/articles/dataset/A_Multiclass_Dataset_for_Real-Time_Fresh_and_Defective_Vegetables/30596084) is a CC BY 4.0 YOLO object-detection dataset. It has seven vegetables (potato, bitter gourd, brinjal, capsicum, onion, pointed gourd, tomato), each labeled fresh or defected. This is a visible quality/object detection task, not the MobileNet freshness model. It does not contain cucumber or egg. The project preparation script verifies the archive checksum and labels, then writes an absolute-path YOLO config and attribution:

```bash
python training/prepare_vegqual.py
python training/train_yolo.py --data dataset/detection/vegqual/data.yaml --epochs 60
```

After training, set `FRESHTRACK_YOLO=models/food_detection/custom_food.pt` in `.env` and restart the app. Preserve `dataset/detection/vegqual/ATTRIBUTION.txt` when using or redistributing the dataset/model.

#### LVIS Fruits and Vegetables: 63 produce classes

The [upstream dataset repository](https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset) describes this 8,221-image, 63-class subset of LVIS, including cucumber and banana; the dataset is also hosted on [Kaggle](https://www.kaggle.com/datasets/henningheyen/lvis-fruits-and-vegetables-dataset). The upstream YAML points validation at the test set; `prepare_lvis_produce` validates labels and writes corrected train, validation and held-out test paths:

```bash
kaggle datasets download henningheyen/lvis-fruits-and-vegetables-dataset \
  -p data/raw/lvis-fruits-vegetables --unzip
python -m training.prepare_lvis_produce
python -m training.train_yolo \
  --data data/raw/lvis-fruits-vegetables/freshtrack_data.yaml \
  --epochs 30 --imgsz 512 --batch 8 \
  --out models/food_detection/freshtrack_lvis.pt
```

Once the checkpoint exists, a restarted app automatically prefers `models/food_detection/freshtrack_lvis.pt`; an explicit `FRESHTRACK_YOLO` environment value still takes precedence. The checkpoint specializes in the dataset's produce classes and may not recognize non-produce foods covered by the COCO model. Evaluate cucumber and banana on the held-out test set with `python -m training.evaluate_lvis_produce`. If you decide to fine-tune further, pass the trained checkpoint using `--weights` and write the result to a new `--out` path. The dataset is a subset of LVIS; preserve its attribution and review upstream image rights before redistribution.

#### Your own detector dataset

For a custom detector, use a dataset with clear reuse terms and bounding boxes for each food class you need. A classification dataset with one class per whole image is not interchangeable with object-detection labels. Convert the selected dataset to YOLO format, separate training and validation images by original source/video (not random near-duplicate frames), and inspect label boxes before training. Include varied fridge lighting, package designs, viewing angles and partly occluded items; evaluate on new images that were not used for training.

#### Optional Roboflow workflow inference

Roboflow is a separate, optional hosted inference path; it is not local training and does not expand the freshness checkpoint. Configure a replacement API key locally using `.env.example`. The photo/live cloud options are opt-in because enabling them sends images to Roboflow and can incur usage charges. Confirm remote detections before saving inventory.

The strawberry example image shared with the project is a results figure, not a labeled dataset. It cannot be used to train a model by itself. Training a strawberry freshness detector would require original strawberry photos and explicit condition labels (and bounding boxes if the result must locate each strawberry).

### Package-date OCR is not currently a trainable project model

The date pipeline uses general-purpose Tesseract OCR plus FreshTrack's date parser. The parser needs legible OCR text; more examples of date strings alone will not teach the app to recognize blurry package images. A custom OCR model would need image crops paired with exact text transcriptions, plus separate tests covering package types, print styles, glare, blur and distance. Date type should still be confirmed from its printed label, and ambiguous dates must remain user-confirmed rather than guessed. The current live and photo flows use the same OCR engine and date parser.

## 3. OCR notes

- Tesseract must be installed separately: `brew install tesseract`.
- The reader cleans the photo (upscale, contrast boost, noise reduction, Otsu and adaptive thresholds), runs Tesseract on each version, and merges the dates it finds.
- It understands EXP / EXPIRY, BEST BEFORE / BEST BY, USE BY, SELL BY, MFG / MANUFACTURED, and PACKED. Each date gets the type of the label printed next to it.
- Numeric dates like `05/06/2027` could mean 5 June or 6 May. The app shows both and makes you choose. It never guesses.
- Photograph labels flat, close and evenly lit. Curved, shiny, embossed or faded print is where OCR fails.
- To swap OCR engines, implement the `OCREngine` class in `modules/expiration_date_reader.py` (two methods) and pass it to `read_expiration_dates`.
