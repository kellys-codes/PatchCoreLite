# ⚡️ PatchCore Lite (Transistor Defect Inspector)

Lightweight and energy-efficient anomaly detection pipeline for detecting defects on transistor images using feature memory + nearest-neighbor matching.

## 📚 Tech stack
- **Language:** Python 3.12
- **Machine Learning/Computer Vision:** PyTorch, TorchVision (ResNet-18), OpenCV, Pillow, NumPy
- **Web UI:** Streamlit
- **Deployment:** Docker

## 🚀 Features
- **Train on good parts only:** learns what a normal transistor looks like from defect-free images, no defect examples needed
- **Automatic threshold:** figures out the normal-vs-anomaly cutoff on its own from the training data
- **Accuracy testing:** run a script to check how well it detects defects on a labeled test set
- **Web app with heatmaps:** upload a photo (or use your camera) and see a highlighted overlay showing exactly where the defect is

## What it does? (high level)
1. Load **ResNet-18** pretrained on ImageNet.
2. Capture intermediate spatial feature maps from `layer2[-1]` and `layer3[-1]` using **forward hooks**.
3. Convert spatial features into a set of patch descriptors and build a **memory bank** (optionally subsampled via a coreset strategy).
4. For each test image, compute its patch descriptors and score them by **nearest-neighbor distance** to the memory bank.
5. Aggregate patch distances into a single image anomaly score using a **log-sum-exp** formulation.
6. Decide **NORMAL vs ANOMALY** using an auto-computed threshold:

> **threshold = mean(train_scores) + 3 × std(train_scores)**

## Data format
The code expects this folder layout:

```
transistor/
  train/good/*.jpg          # normal images only
  test/good/*.jpg            # normal images
  test/<defect_name>/*.jpg   # one folder per defect type
```

## Setup & Usage
```bash
git clone <your-repo-url>
cd PatchCoreLite
pip install -r requirements.txt

python train.py        # build the memory bank and auto-threshold -> transistor_lite_model.pt
python test.py         # evaluate accuracy on transistor/test/...
streamlit run app.py   # launch the web app for upload + heatmap inference
```

## Configuration
A couple of constants worth knowing about (set at the top of `train.py` / `app.py` / `test.py`):
- `CORESET_PCT`: how much of the memory bank to keep. Higher = more accurate but slower/bigger.
- `TEMPERATURE`: how sensitive the anomaly score is. Lower = more sensitive, but noisier.

## Troubleshooting
- **Model not found:** run `python train.py` first.
- **Dataset errors:** check that `transistor/train/good` and `transistor/test/...` exist.
- **Poor accuracy:** try raising `CORESET_PCT` or lowering `TEMPERATURE`.

## Docker (optional)
```bash
docker build -t patchcore-lite .
docker run -p 7860:7860 patchcore-lite
```

## 🔗 Links to Demo
- Video: https://drive.google.com/file/d/1nMGnkIKF9_AHRZ_Ykq0Xy9rFuhpk8bNX/view?usp=sharing
- Live app: https://kell08-transistor-inspector.hf.space (may take a minute to load)
