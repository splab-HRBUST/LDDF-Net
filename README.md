# LDDF-Net

## 1. Start Up

### Prepare the Environment

```bash
conda activate yourEnv
pip install -r requirements.txt
```

### Get the Pre-trained Model

```bash
python get_model.py
```

## 2. Finetune Whisper

### 1. ESC-50 Experiment

```bash
cd src/whisper_at_train/esc-50/
./run_esc.sh
```

### 2. AudioSet20k Experiment

```bash
cd src/whisper_at_train/audioset20k/
./run_audioset20k.sh
```

## 3. Noise-Robust ASR Experiment

### 1. Extract Intermediate Features

```bash
cd src/noise_robust_asr/intermediate_feat_extract/
python extract_esc_whisper.py
```

## 4. Frozen Whisper Experiment

Download the Whisper feature of ESC-50 from [https://www.dropbox.com/s/hmmdopfjlq3o3vs/esc_feat.zip?dl=1](https://www.dropbox.com/s/hmmdopfjlq3o3vs/esc_feat.zip?dl=1).

```bash
cd src/whisper_at_train/esc-50/
./run_esc.sh
```

**Note:** You need to replace lines 84-89 in `run_esc.sh` with your dataset json path, label.csv and feature path. The datapath json and label.csv are ready for you in the project.
