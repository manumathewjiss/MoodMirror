# Source Code Directory

This directory contains Python scripts (`.py` files) for production-ready code.

## Directory Structure

```
src/
├── data_preparation/     (Data loading & preprocessing)
├── model_training/       (Training scripts)
├── evaluation/           (Evaluation & metrics)
└── deployment/           (FastAPI application)
```

## Purpose

While Jupyter notebooks (`.ipynb`) are great for exploration and showing your work, Python scripts (`.py`) are better for:
- Reusable code
- Production deployment
- Cleaner organization
- Easier testing

## Workflow

1. **Explore in notebooks** (`notebooks/` folder)
   - Experiment with different approaches
   - Document your process
   - Show results inline

2. **Extract to scripts** (`src/` folder)
   - Once you know what works
   - Create clean, reusable functions
   - Remove exploratory code

## What Goes Where

### `data_preparation/`
Scripts for data handling:
- `load_data.py` - Load datasets
- `preprocess.py` - Text cleaning functions
- `utils.py` - Helper functions

### `model_training/`
Scripts for training models:
- `train_svm.py` - Train baseline SVM
- `train_bert.py` - Train DistilBERT
- `config.py` - Training configuration

### `evaluation/`
Scripts for model evaluation:
- `evaluate.py` - Calculate metrics
- `visualize.py` - Create plots
- `compare_models.py` - Model comparison

### `deployment/` ⭐ Most Important
FastAPI application for live demo:
- `app.py` - Main FastAPI app
- `model_loader.py` - Load trained models
- `schemas.py` - API input/output formats

## For This Project

**Priority**: Focus on `deployment/` folder
- This is what you'll demo during presentation
- Other folders are optional (nice to have)
- Notebooks are sufficient for academic requirements

## Note

Creating scripts in `data_preparation/`, `model_training/`, and `evaluation/` is **optional** for this academic project. Your Jupyter notebooks already contain all the necessary code.

The **deployment/** folder is essential for the live API demo.

