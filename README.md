# FreshTrack AI

An intelligent agent that watches your fridge and pantry. It identifies food from photos, judges how fresh fruit looks, reads dates off packaging, remembers what you own, and tells you **what to eat first**.

Built for the Intelligent Agents final project. Utility-based agent with learning components.

| Part | What it does | Technology |
|---|---|---|
| Food detection | Photo scans combine a 63-class produce detector with a local open-vocabulary model prompted for meat, dairy and other foods; live camera uses the produce detector | YOLOv8 + YOLO-World; optional custom YOLO or Roboflow workflow |
| Freshness | Fresh / Aging / Visible spoilage for foods listed by the trained checkpoint (the shipped baseline covers apple, banana, orange) | MobileNetV2, transfer learning |
| Date reader | Reads EXP, BEST BEFORE, USE BY, SELL BY, MFG, PACKED dates | OpenCV + Tesseract OCR |
| Inventory | Persistent fridge and pantry records, six food categories | SQLite |
| Decision engine | Scores how urgently each item needs attention | scikit-fuzzy |
| Alerts | Past date / due today / coming up, per date type | Pure Python |
| Interface | Home, live camera scan, photo scan, Fridge, About | Streamlit + WebRTC |

## Run it (macOS)

```bash
# 1. Open this folder in VS Code, then in its terminal:
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. OCR needs the Tesseract program (pip only installs the Python wrapper)
brew install tesseract

# 3. Start the app
python -m streamlit run app.py
```

The app opens at http://localhost:8501. It works right away: the repository already includes a trained baseline freshness model (see "Freshness model" below). Detector weights are downloaded on first use; the extra local photo-scan model is about 25 MB.

If `brew` is missing, install Homebrew from https://brew.sh first. If Tesseract installs somewhere unusual, set `TESSERACT_CMD` in `.env` (see `.env.example`).

## Using it

1. **Scan, Food photo**: use Live camera for continuous detection (the local detector checks frames about twice per second) and freshness assessment of supported fruits and vegetables, or upload/capture a still photo. The confirmation form stays beside the video. Confirm the suggested item before it is added to the fridge. A small photo preview is saved with scanned items and shown in **Fridge → Manage**. A confident produce freshness result is saved and shown on its fridge card and under **Manage → Freshness info**. Other food categories do not show freshness info.
2. **Scan, Package date**: choose Live camera to read a label continuously or Photo or upload for a still image. Pick which item it belongs to, check each date the scanner found, and confirm. Only confirmed dates feed alerts.
3. **Fridge**: search, filter, and manage everything. Add things by hand, edit quantities, add or confirm dates.
4. **Home**: the "Use first" shelf ranks what to eat first. "Check before eating" holds spoiled-looking food and anything past a date.

Repeated detections of the same food in a photo are grouped into one item, with the detected count prefilled as its editable quantity. **Added to fridge** is the date the inventory record was created; a package date is the expiration or best-before date printed on the food, not the photo's capture date.

On the Fridge page, an item's category, quantity, storage location, notes, and **Added to fridge** date can be edited under Manage. Hand-entered package dates are labeled as expiration or best-before dates. **Mark consumed** and **Delete item** both remove an item from the active fridge and keep it in Fridge history with the corresponding status; history entries can be restored or permanently deleted. Recent scan dates can be edited on Home.

Recent scans on Home expand to show their saved photo when one is available. New photo and live-camera food scans, and package-date photo/live-camera reads, save a compact preview alongside the scan record. Older food scans can show the linked fridge-item photo when available.

## Add more foods and train models

### Expand object detection with VegQual

VegQual is a CC BY 4.0 dataset with seven vegetables labeled as fresh or visibly defected. It is an object-detection dataset, not a general freshness classifier; it does not add cucumber or egg. The prepared files and attribution are in `dataset/detection/vegqual/`.

```bash
python training/prepare_vegqual.py
python training/train_yolo.py --data dataset/detection/vegqual/data.yaml --epochs 60
```

After training, set `FRESHTRACK_YOLO=models/food_detection/custom_food.pt` in your local `.env`, then restart the app. Keep the generated `ATTRIBUTION.txt` with any use or redistribution of the dataset/model.

### Train broad fruit and vegetable detection

The [Fruits-And-Vegetables-Detection-Dataset repository](https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset) describes this MIT-listed LVIS subset: 8,221 images across 63 classes, including cucumber and banana. It is about 1.4 GB compressed. Retain the dataset attribution and review upstream image terms before redistribution.

```bash
kaggle datasets download henningheyen/lvis-fruits-and-vegetables-dataset \
  -p data/raw/lvis-fruits-vegetables --unzip
python -m training.prepare_lvis_produce
python -m training.train_yolo \
  --data data/raw/lvis-fruits-vegetables/freshtrack_data.yaml \
  --epochs 30 --imgsz 512 --batch 8 \
  --out models/food_detection/freshtrack_lvis.pt
```

The preparation step validates the annotations and corrects the upstream YAML, which points validation at the test set. It keeps the 180-image test split separate. Once training completes, a restarted app automatically prefers `models/food_detection/freshtrack_lvis.pt`; an explicit `FRESHTRACK_YOLO` setting still takes precedence. The checkpoint specializes in the dataset's fruit and vegetable classes; it may not retain non-produce classes from the original COCO detector.

Photo scans also run a local YOLO-World model with prompts for hot dogs, bacon, turkey, beef, chicken, pork, sausage, ham, fish, tuna, shrimp, crab, lobster, eggs, milk, cheese, yogurt, butter, cream, ice cream, bread, pasta, noodles, rice, cereal, beans, tofu, pizza, sandwiches, cake and donuts. Its weights download once if missing (about 25 MB); photos stay on this device unless you explicitly opt into Roboflow cloud inference. This open-vocabulary model is not custom-trained on FreshTrack meat or dairy photos, so confirm every suggestion before adding it. Live camera uses the configured primary detector (produce-focused by default), not this broader prompt model.

The app's default freshness checkpoint is `mobilenetv2_freshness.pt`; the 3,294-image result below is for a separate checkpoint (`mobilenetv2_fresh_stale.pt`), not the default currently configured in the app. Evaluate it without overwriting another model's results:

```bash
python -m training.evaluate_freshness \
  --data dataset/freshness_stale_expanded \
  --ckpt models/freshness_classification/mobilenetv2_fresh_stale.pt
```

By default, the report and confusion matrix are saved in a checkpoint-specific `*_evaluation/` directory. This separate checkpoint scored 98.3% accuracy and 0.965 macro-F1 on 3,294 held-out dataset images; spoiled-class precision was 0.908 and recall 0.927 (245 examples). These are dataset results, not household-photo results. The test split has no spoiled examples for bitter gourd, capsicum, or tomato, so do not claim spoiled-condition performance for those foods. The checkpoint embeds a creation timestamp later than the stated submission deadline; verify provenance before presenting these results as pre-deadline work. The default checkpoint and its older 1,246-image report also have post-deadline embedded/report timestamps.

The produce detector's held-out evaluation is saved in [freshtrack_lvis_test_metrics.json](models/food_detection/freshtrack_lvis_test_metrics.json): 180 images, mAP50 0.1815, mAP50-95 0.1088; cucumber AP50 0.2667 and banana AP50 0.8336. This shows why merely adding classes or training epochs is not enough; evaluate per-class performance and collect more representative labeled data. No household photos or custom meat/dairy detection labels are currently included in this workspace.

Evaluate cucumber and banana on the held-out test split (do not use that split for training):

```bash
python -m training.evaluate_lvis_produce
```

If test results show the model needs further fine-tuning, start a separate run from the trained checkpoint instead of overwriting it:

```bash
python -m training.train_yolo \
  --data data/raw/lvis-fruits-vegetables/freshtrack_data.yaml \
  --weights models/food_detection/freshtrack_lvis.pt \
  --epochs 30 --imgsz 512 --batch 8 \
  --out models/food_detection/freshtrack_lvis_more.pt
```

### Expand visual freshness with Freshness44

The Kaggle Freshness44 dataset is about 7.4 GB, licensed CC BY-SA 4.0, and contains 53,616 fresh/rotten images across 22 produce types. The download needs your own Kaggle account. Install/configure Kaggle locally; never send Kaggle credentials in chat or commit them.

```bash
python -m pip install kaggle
python -m training.prepare_dataset
kaggle datasets download -d siavash93/freshness44 -p data/raw/freshness44 --unzip
python -m training.prepare_freshness44
python -m training.check_freshness_dataset --data dataset/freshness_expanded
python -m training.train_freshness --data dataset/freshness_expanded
python -m training.evaluate_freshness --data dataset/freshness_expanded
```

The preparation script preserves the existing three-level fruit data (`fresh`, `aging`, `spoiled`) and maps Freshness44's `fresh`/`rotten` folders to `fresh`/`spoiled`. It writes source attribution and a manifest used to list trained foods in the checkpoint. Freshness44's CC BY-SA terms require attribution and share-alike compliance for adapted material; review the license before distributing a resulting model.

### Add the Fresh and Stale produce dataset

Kaggle lists [Fresh and Stale Images of Fruits and Vegetables](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables) as CC0 and about 1.5 GB. It covers apples, bananas, bitter gourd, capsicum, oranges, and tomatoes. Download it locally with your Kaggle account; never share Kaggle credentials. The dataset author says some images came from another dataset and some were extracted from videos, so the preparation script checks exact duplicates against the base set and keeps perceptual near-duplicate clusters in one split. This reduces, but cannot eliminate, leakage risk.

```bash
python -m pip install kaggle
kaggle datasets download -d raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables -p data/raw/fresh-stale --unzip
python -m training.prepare_dataset
python -m training.prepare_fresh_stale
python -m training.check_freshness_dataset --data dataset/freshness_stale_expanded
python -m training.train_freshness --data dataset/freshness_stale_expanded
python -m training.evaluate_freshness --data dataset/freshness_stale_expanded
```

The source's **stale** label maps to FreshTrack's **aging** class—not to spoiled. This is an approximate label mapping, not a medical or safety judgment. The output includes attribution/provenance notes; although Kaggle lists CC0, review the source's acknowledgment of reused images before redistributing data or a trained model.

To use both extra datasets in one training set, prepare the stale dataset first, then merge Freshness44 on top:

```bash
python -m training.prepare_freshness44 --base dataset/freshness_stale_expanded --out dataset/freshness_all
python -m training.check_freshness_dataset --data dataset/freshness_all
python -m training.train_freshness --data dataset/freshness_all \
  --out models/freshness_classification/fresh44/mobilenetv2_fresh44.pt
python -m training.evaluate_freshness --data dataset/freshness_all \
  --ckpt models/freshness_classification/fresh44/mobilenetv2_fresh44.pt
```

After both custom models are trained, run FreshTrack with them using:

```bash
FRESHTRACK_YOLO=models/food_detection/freshtrack_lvis.pt \
FRESHTRACK_FRESHNESS_CKPT=models/freshness_classification/fresh44/mobilenetv2_fresh44.pt \
python -m streamlit run app.py --server.port 8505
```

The Fridge page shows a saved visual freshness label and model confidence for produce under **Manage → Freshness info**. New estimates are saved when supported fruits or vegetables are scanned and confirmed; dairy and other categories do not show freshness info. Food-photo and live-camera scans also save a small preview with newly added items. Previously saved items without a stored photo will not have a preview until scanned again.

The shipped `models/freshness_classification/mobilenetv2_freshness.pt` is a **baseline** trained in a small sandbox (CPU only, 128 px images, short training). It scored **96.9% accuracy and 0.970 macro-F1 on 1,246 held-out test photos** (`models/freshness_classification/metrics.json`). Those research-dataset scores are not a guarantee of real-world performance. For evaluation on separate household photos, see [documentation/MODEL_TRAINING_GUIDE.md](documentation/MODEL_TRAINING_GUIDE.md).

### Optional Roboflow workflow

An opt-in workflow can add classes such as cucumber and egg to photo and live-camera scans. Configure the workspace, workflow ID, and a rotated API key in local `.env` using [.env.example](.env.example). The scan-page cloud options are unchecked by default: enabling them sends images to Roboflow and may incur usage charges. Never commit or share the API key. Remote inference only suggests detections; confirm them before adding inventory. The workflow expands food detection, not freshness classification.

## Run the tests

```bash
python -m pytest
```

Slow integration tests (real OCR) skip themselves if Tesseract is missing.

## Project layout

```
app.py                  entry point
config/settings.py      paths, thresholds, categories
ui/                     Streamlit pages and theme (home, scan, fridge, about)
modules/                detector, freshness classifier, OCR reader, fuzzy engine,
                        alerts, recommendations, inventory, database
database/schema.sql     SQLite tables
training/               dataset prep, training, evaluation, optional YOLO training
models/                 checkpoints and evaluation results
assets/styles.css       the look of the app
tests/                  pytest suite
documentation/          architecture and training guides
```

## Honest limits

- **It judges how food looks, not whether it is safe.** It cannot see bacteria or toxins, and "Fresh" is never proof of safety.
- The shipped freshness baseline supports **apple, banana, orange**. Retraining on Freshness44 adds the foods present in that dataset to the model's supported list, but its labels describe visual appearance only and are not safety judgments.
- Photo scans use a produce detector plus an open-vocabulary local model for additional foods. The live camera remains produce-focused unless configured with a custom detector or optional Roboflow workflow. Extra food suggestions are not FreshTrack-trained and should be confirmed; detection support does not mean freshness support.
- Live camera detection runs only while the browser camera is started and the Scan page is open. It is not unattended, always-on fridge monitoring.
- OCR fails on curved, shiny, faded or blurry packaging. Ambiguous dates like `05/06/2027` are never guessed. You choose.
- A manufacturing or packing date is never turned into an expiry date.
- The fuzzy priority is a ranking, not a safety verdict. Visibly spoiled food is never recommended for eating.
- The shipped test scores come from a public research dataset with clean, centered photos. Photos from your own fridge usually score lower.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Tesseract is not installed" | `brew install tesseract`, restart the app |
| `command not found: python3.11` | `brew install python@3.11` |
| First photo scan is slow | YOLO downloads its weights once, and models load on first use |
| Port already in use | `python -m streamlit run app.py --server.port 8502` |
| Freshness says "not been trained" | The checkpoint is missing: run the training steps above |
