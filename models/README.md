# Models Directory

This directory stores trained models.

## Models Generated

### 1. Baseline Model (SVM)
```
models/
├── svm_baseline.pkl        (~5 MB)
└── tfidf_vectorizer.pkl    (~20 MB)
```

**Files:**
- `svm_baseline.pkl`: Trained SVM classifier
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer (must use same one for predictions)

**How to load:**
```python
import joblib
model = joblib.load('models/svm_baseline.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
```

### 2. Deep Learning Model (DistilBERT)
```
models/
└── distilbert_emotion_classifier/
    ├── config.json
    ├── pytorch_model.bin       (~250 MB)
    ├── tokenizer_config.json
    ├── vocab.txt
    └── special_tokens_map.json
```

**How to load:**
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained('models/distilbert_emotion_classifier')
tokenizer = DistilBertTokenizer.from_pretrained('models/distilbert_emotion_classifier')
```

## ⚠️ Important Notes

### File Size
- SVM models: ~25 MB total
- DistilBERT model: ~250 MB total

**These are too large for GitHub!**

### For Submission

**Option 1**: Include models in ZIP (if under 500MB limit)

**Option 2**: Upload to cloud storage
- Google Drive
- Dropbox
- GitHub Releases (for large files)
- Include download link in main README

### For Team Sharing

Share models via:
- Google Drive shared folder
- Cloud storage
- Or re-train (notebooks have all code)

## Model Checkpoints

During training, checkpoints may be saved in:
```
models/
└── distilbert_checkpoints/
    └── checkpoint-XXX/
```

These can be deleted after training completes.

