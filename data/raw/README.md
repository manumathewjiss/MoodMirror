# Raw Data Directory

This directory contains the original, unmodified datasets.

## Dataset: GoEmotions

**Source**: Google Research  
**Size**: ~58,000 text samples  
**Emotions**: 7 categories (joy, sadness, anger, fear, surprise, disgust, neutral)

## How to Download

The dataset will be downloaded using the `00_download_dataset.ipynb` notebook.

Alternatively, you can download manually from:
- **HuggingFace**: https://huggingface.co/datasets/go_emotions
- **Kaggle**: Search for "GoEmotions" dataset

## Expected Files

After running the download notebook, you should have:
```
raw/
├── train_raw.csv       (~43,000 samples)
├── val_raw.csv         (~5,400 samples)
└── test_raw.csv        (~5,400 samples)
```

## File Format

Each CSV file contains:
- `text`: The text content
- `emotion`: The emotion label (joy, sadness, anger, fear, surprise, disgust, neutral)

## ⚠️ Important

**DO NOT modify these files!** They are your backup.  
All preprocessing should create new files in `data/processed/`

