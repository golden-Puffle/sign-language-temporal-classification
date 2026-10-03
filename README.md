# Sign Language Recognition using Temporal Classification

UE24CS352A Machine Learning mini-project (team of two).

We recognise isolated sign-language words from video by extracting body and hand landmarks with MediaPipe and classifying the resulting time series with temporal models.

## Team

| Member | Role |
|---|---|
| Monisha | Data and preprocessing: dataset preparation, landmark extraction, normalisation, sequence generation, labelling, dataset analysis |
| Nadeem | Model and prediction: temporal classification architecture, training, tuning, evaluation, prediction/demo |

## Dataset

[WLASL (Word-Level American Sign Language)](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) from Kaggle. We use the 25 signs with the most available videos.

The raw videos and processed arrays are **not stored in this repo** (too large). To reproduce:

1. Download WLASL from the Kaggle link above and place it in `data/raw/`, or run the notebook directly on Kaggle with the dataset attached.
2. Run `notebooks/01_data_preprocessing.ipynb`. It produces `wlasl25_landmarks.npz`, `label_map.json`, `config.json` and `metadata.csv`.
3. Place those files in `data/processed/`.

Processed data (ready to use): `https://drive.google.com/drive/folders/1yMTiisowuMznuah1i9NfMWH5CF0-JjQ8?usp=sharing`

## Preprocessing pipeline

1. **Sign selection:** the 25 signs with the most videos on disk.
2. **Landmark extraction:** MediaPipe Holistic gives 6 upper-body pose points plus 21 points per hand, with x, y, z each, for 144 features per frame.
3. **Quality filter:** videos where the body or hands are rarely detected are dropped.
4. **Spatial normalisation:** coordinates are made relative to the shoulder mid-point and divided by shoulder width.
5. **Missing values:** undetected landmarks are linearly interpolated over time.
6. **Temporal resampling:** every sequence is resampled to 40 frames.
7. **Split:** train / validation / test using the official WLASL split.
8. **Augmentation (training only):** scale, rotation, shift, jitter and speed change.

Output format: `X` has shape `(num_samples, 40, 144)`, `y` holds integer labels (see `label_map.json`).

## Setup

Use Python 3.10 to 3.12.

```bash
git clone https://github.com/<username>/sign-language-temporal-classification.git
cd sign-language-temporal-classification
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Loading the data

```python
import numpy as np, json

d = np.load("data/processed/wlasl25_landmarks.npz")
X_train, y_train = d["X_train"], d["y_train"]
X_val,   y_val   = d["X_val"],   d["y_val"]
X_test,  y_test  = d["X_test"],  d["y_test"]
# d["X_train_aug"], d["y_train_aug"] = training set plus augmented copies

label_map = json.load(open("data/processed/label_map.json"))   # sign name -> integer
```

## Running the model on a new video (demo)

```python
from src.preprocessing import preprocess_video

x = preprocess_video("my_clip.mp4")   # shape (1, 40, 144), same format as training data
# prediction = model.predict(x)
```

## Repository structure

```
data/
  raw/                 # original videos (not tracked by git)
  processed/           # arrays produced by the preprocessing notebook (not tracked)
models/                # trained models
notebooks/             # data preprocessing and analysis notebooks
src/
  preprocessing.py     # landmark extraction, normalisation, resampling, augmentation
figures/               # plots used in the report
requirements.txt
```

## Results

To be added once models are trained.