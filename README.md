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

### 1. Start Training

```bash
cd src/whisper_at_train/
./run_as_full_train.sh
```

### 2. ESC-50 Experiment

```bash
cd src/whisper_at_train/esc-50/
./run_esc.sh
```

### 3. AudioSet20k Experiment

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

### 2. Run ASR Experiments

```bash
cd src/noise_robust_asr/asr_experiments/
python transcribe_whisper.py
python transcribe_hubert_large.py
python transcribe_wav2vec_base.py
python transcribe_wav2vec_robust.py
python transcribe_esc_hubert_xl.py
```

### 3. Generate Noisy Speech

```bash
cd src/noise_robust_asr/asr_experiments/
python gen_noisy_speech.py
```

### 4. Compute WER

```bash
cd src/noise_robust_asr/asr_experiments/
python compute_wer.py
python compute_wer_cla.py
```

## 4. Frozen Whisper Experiment

Download the Whisper feature of ESC-50 from [https://www.dropbox.com/s/hmmdopfjlq3o3vs/esc_feat.zip?dl=1](https://www.dropbox.com/s/hmmdopfjlq3o3vs/esc_feat.zip?dl=1).

```bash
cd src/whisper_at_train/esc-50/
./run_esc.sh
```

**Note:** You need to replace lines 84-89 in `run_esc.sh` with your dataset json path, label.csv and feature path. The datapath json and label.csv are ready for you in the project.

## Project Structure

```
LDDF-Net/
├── package/
│   └── whisper-at/          # Whisper-AT package
├── pretrained_models/       # Pre-trained models directory
├── src/
│   ├── noise_robust_asr/    # Noise-robust ASR experiments
│   │   ├── asr_experiments/
│   │   ├── intermediate_feat_extract/
│   │   └── plots/
│   └── whisper_at_train/    # Whisper-AT training scripts
│       ├── audioset20k/
│       ├── esc-50/
│       ├── log/
│       └── utilities/
├── get_model.py             # Pre-trained model download script
├── requirements.txt         # Dependencies
└── test.py                  # Test script
```

## Requirements

- Python 3.10+
- PyTorch
- Whisper-AT
- Other dependencies listed in requirements.txt
