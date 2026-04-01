# Depression-dataset
# Multimodal Depression Detection: Audio-Text Fusion Evaluation

This repository provides training code for automatic depression detection based on audio and text modalities. It supports both **early fusion** (feature concatenation → SVM / BiLSTM) and **cross-modal attention fusion** (MulT) strategies. The MulT code is borrowed largely from https://github.com/yaohungt/Multimodal-Transformer. Thank them!

> The audio and text feature extraction scripts are in `feat_extract/`, and all training scripts are in `train/`.

---

## Project Structure

```
├── feat_extract/
│   ├── extract_audio.py     # Extract audio features (logmel / wav2vec2 / hubert / emotion2vec)
│   └── extract_text.py      # Extract MacBERT text features (mean / layer4 / seq)
├── train/
│   ├── train_audio.py       # Audio-only baseline
│   ├── train_text.py        # Text-only baseline
│   └── train_multi.py       # Audio-text fusion (early: SVM / BiLSTM; multi: MulT)
└── README.md
```

---

## Data

### Format

The corpus consists of semi-structured interview recordings. Each recording is segmented into clips, and each clip corresponds to one sample. You need to prepare two split files:

```
├── train.txt
└── val.txt
```

Each line in the `.txt` file follows the format:

```
<relative_audio_path> <label>
```

- `<relative_audio_path>`: path to the `.wav` file relative to the data root, e.g. `root/subj01/seg001.wav`
- `<label>`: `0` for HC (healthy control), `1` for DP (depressed)


### Data Access

The data are protected. Please download and send the signed EULA to edu.cn for access request.



## Feature Extraction
Each script documents its arguments at the top of the file. For example, run `python feat_extract/extract_audio.py` for the audio feature extraction.


## Evaluation Metrics

All models report the following metrics on the validation set after every epoch:

- Per-class Precision / Recall / F1 (HC and DP separately)
- Macro-average Precision / Recall / F1

The best checkpoint is selected by **macro-F1**.

