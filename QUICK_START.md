# 🚀 Quick Start Guide

Welcome to MoodMirror! Follow these steps to get started.

---

## ✅ Step 1: Setup Environment (10 minutes)

### 1.1 Open Terminal

Navigate to project folder:
```bash
cd "/Users/manumathewjiss/Documents/Darsh NLP project"
```

### 1.2 Create Virtual Environment

```bash
python3 -m venv venv
```

### 1.3 Activate Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### 1.4 Install Dependencies

```bash
pip install -r requirements.txt
```

⏱️ This takes 10-15 minutes (downloading ~2GB of libraries)

☕ Grab a coffee while it installs!

---

## ✅ Step 2: Download Dataset (20 minutes)

### 2.1 Open Jupyter Notebook

```bash
jupyter notebook
```

This opens in your browser.

### 2.2 Create First Notebook

1. Navigate to `notebooks/` folder
2. Click "New" → "Python 3"
3. Save as `00_download_dataset.ipynb`

### 2.3 Download GoEmotions

Copy and run this code cell by cell:

**Cell 1:**
```python
# Install datasets library
!pip install datasets
```

**Cell 2:**
```python
from datasets import load_dataset
import pandas as pd
import os

print("Downloading GoEmotions dataset...")
dataset = load_dataset("go_emotions", "simplified")
print("✅ Download complete!")
```

**Cell 3:**
```python
# Check what we got
print(f"Train: {len(dataset['train'])} samples")
print(f"Validation: {len(dataset['validation'])} samples")
print(f"Test: {len(dataset['test'])} samples")
```

**Cell 4:**
```python
# Convert to DataFrames
train_df = pd.DataFrame(dataset['train'])
val_df = pd.DataFrame(dataset['validation'])
test_df = pd.DataFrame(dataset['test'])

# Show first few rows
print(train_df.head())
```

Continue following the full code in the detailed pipeline document.

---

## ✅ Step 3: Next Steps

Once setup is complete:

### Your Tasks (Team Member 1):
1. ✅ Complete `01_data_exploration.ipynb`
2. ✅ Complete `02_data_preprocessing.ipynb`
3. ✅ Complete `03_baseline_model.ipynb`

### Teammate Tasks (Team Member 2):
1. ✅ Complete `04_deep_learning_model.ipynb` (recommend using Google Colab with GPU!)
2. ✅ Complete `05_model_comparison.ipynb`

### Both Together:
1. ✅ Build FastAPI deployment
2. ✅ Write report
3. ✅ Create presentation

---

## 📚 Helpful Commands

### Activate/Deactivate Environment

**Activate:**
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

**Deactivate:**
```bash
deactivate
```

### Start Jupyter

```bash
jupyter notebook
```

### Run FastAPI (later)

```bash
cd src/deployment
uvicorn app:app --reload
```

---

## 🆘 Troubleshooting

### "Command not found: python3"
Try `python` instead of `python3`

### "Permission denied"
On macOS, you might need:
```bash
chmod +x venv/bin/activate
```

### "Module not found"
Make sure virtual environment is activated (you see `(venv)` in terminal)

### "Jupyter not found"
```bash
pip install jupyter notebook
```

### PyTorch Installation Issues
If torch doesn't install properly:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 📖 Documentation

- **Full Pipeline**: See the detailed explanation document
- **README.md**: Project overview
- **data/raw/README.md**: Dataset information
- **models/README.md**: Model information
- **src/README.md**: Code organization

---

## ✅ Checklist

Setup Phase:
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Jupyter notebook running
- [ ] Dataset downloaded

Ready to Code! 🎉

---

**Need Help?** Ask questions as you go! Understanding is more important than speed.

