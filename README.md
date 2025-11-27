# MoodMirror: Emotion Detection from Text using NLP

**NLP Final Project - Academic Implementation**

---

## 📋 Project Overview

**MoodMirror** is an AI-powered emotion detection system that analyzes text to identify and classify human emotions. This project demonstrates the complete Machine Learning pipeline from data collection to model deployment.

### **Problem Statement**
Understanding emotions in text is crucial for applications like mental health monitoring, customer feedback analysis, and social media sentiment tracking. This project builds a multi-class emotion classifier that can detect 7 distinct emotions from user-written text.

### **Emotions Detected**
- Joy
- Sadness
- Anger
- Fear
- Surprise
- Disgust
- Neutral

---

## 🎯 Project Objectives

1. **Data Collection**: Gather and prepare emotion-labeled text dataset
2. **Data Preprocessing**: Clean, tokenize, and prepare text for training
3. **Model Training**: Train multiple ML/DL models for emotion classification
4. **Model Evaluation**: Compare models using accuracy, precision, recall, F1-score
5. **Deployment**: Build REST API for real-time emotion detection
6. **Analysis**: Provide insights and recommendations for improvement

---

## 📁 Project Structure

```
MoodMirror/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned and preprocessed data
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA and visualization
│   ├── 02_data_preprocessing.ipynb     # Data cleaning and preparation
│   ├── 03_baseline_models.ipynb        # Traditional ML models
│   ├── 04_deep_learning_models.ipynb   # LSTM, BERT fine-tuning
│   └── 05_model_evaluation.ipynb       # Comparison and analysis
├── src/
│   ├── data_preparation/       # Data loading and preprocessing scripts
│   ├── model_training/         # Training scripts for different models
│   ├── evaluation/             # Evaluation metrics and visualization
│   └── deployment/             # FastAPI application for deployment
├── models/                     # Saved trained models
├── results/
│   ├── figures/                # Plots and visualizations
│   ├── metrics/                # Performance metrics (JSON/CSV)
│   └── models/                 # Model comparison results
├── tests/                      # Unit tests
├── docs/
│   ├── report.pdf              # Academic report
│   └── presentation.pptx       # Final presentation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Getting Started

### **Prerequisites**
- Python 3.8+
- pip or conda
- Jupyter Notebook
- Git

### **Installation**

1. **Clone the repository**
```bash
cd "Darsh NLP project"
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset** (instructions in notebooks/01_data_exploration.ipynb)

---

## 📊 Dataset

**Primary Dataset**: GoEmotions Dataset (Google Research)
- **Size**: 58,000+ carefully curated text samples
- **Source**: Reddit comments
- **Labels**: 27 emotion categories (we'll map to 7 main emotions)
- **Split**: 80% train, 10% validation, 10% test

**Alternative/Supplementary Datasets**:
- Emotion Dataset for NLP (Kaggle)
- DailyDialog Dataset
- EmoContext (SemEval 2019)

---

## 🧪 Methodology

### **Phase 1: Data Preparation (Teammate 1)**
- Data collection and exploration
- Text cleaning and preprocessing
- Train/validation/test split
- Data augmentation (optional)

### **Phase 2: Model Development (Both)**
- **Baseline Models**:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
  
- **Deep Learning Models**:
  - LSTM with Word2Vec embeddings
  - BiLSTM with attention
  - DistilBERT (fine-tuned)

### **Phase 3: Evaluation (Teammate 2)**
- Performance metrics calculation
- Confusion matrix analysis
- Error analysis
- Model comparison

### **Phase 4: Deployment (Both)**
- FastAPI REST API
- Model serving
- Testing and documentation

---

## 📈 Expected Results

### **Performance Targets**
- Baseline Model: 60-70% accuracy
- Deep Learning Models: 80-90% accuracy
- Final Model: 85%+ accuracy with good F1-scores across all classes

### **Deliverables**
1. ✅ Complete codebase (Jupyter notebooks + Python scripts)
2. ✅ Trained models (saved in `models/` directory)
3. ✅ Evaluation report with metrics and visualizations
4. ✅ REST API for real-time predictions
5. ✅ Academic report (PDF)
6. ✅ Presentation slides (PPTX)
7. ✅ Dataset (included or download links)

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Data Processing** | pandas, numpy, nltk, spaCy |
| **Visualization** | matplotlib, seaborn, plotly |
| **ML Models** | scikit-learn |
| **Deep Learning** | PyTorch, Transformers (HuggingFace) |
| **API** | FastAPI, uvicorn |
| **Experiment Tracking** | Weights & Biases / MLflow (optional) |

---

## 👥 Team & Task Division

### **Team Member 1: [Your Name]**
**Responsibilities**:
- Data collection and exploration (Notebook 01)
- Data preprocessing pipeline (Notebook 02)
- Baseline model implementation (Notebook 03)
- API deployment (src/deployment)
- README and documentation

### **Team Member 2: [Teammate Name]**
**Responsibilities**:
- Deep learning models (Notebook 04)
- Model evaluation and comparison (Notebook 05)
- Visualization and results analysis
- Academic report writing
- Presentation slides creation

**Collaborative Tasks**:
- Code review and testing
- Results discussion
- Final presentation preparation
- ZIP file compilation for submission

---

## 📝 Usage

### **Training Models**

```bash
# Train baseline models
python src/model_training/train_baseline.py

# Train LSTM model
python src/model_training/train_lstm.py

# Fine-tune BERT
python src/model_training/train_bert.py
```

### **Evaluation**

```bash
# Evaluate all models
python src/evaluation/evaluate_models.py

# Generate comparison report
python src/evaluation/generate_report.py
```

### **Running the API**

```bash
# Start FastAPI server
cd src/deployment
uvicorn app:app --reload

# API will be available at: http://localhost:8000
# Documentation at: http://localhost:8000/docs
```

### **API Example**

```bash
# Predict emotion
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so excited about this project!"}'

# Response:
{
  "emotion": "joy",
  "confidence": 0.92,
  "all_probabilities": {
    "joy": 0.92,
    "surprise": 0.05,
    "neutral": 0.02,
    "sadness": 0.01
  }
}
```

---

## 📊 Results Preview

### Baseline Model Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression (Baseline) | 53.66% | 61.79% | 53.66% | 55.56% |
| DistilBERT | TBD | TBD | TBD | TBD |

**Note:** Baseline model uses TF-IDF features with mean-based class balancing (SMOTE + RandomUnderSampler).

---

## 🔍 Key Insights & Improvements

*(Will be added after analysis)*

### **Challenges Faced**
- TBD

### **What Worked Well**
- TBD

### **Future Improvements**
- Multi-modal emotion detection (text + audio + video)
- Real-time emotion tracking dashboard
- Personalized emotion analysis
- Explainable AI for emotion predictions

---

## 📚 References

1. GoEmotions Dataset: https://github.com/google-research/google-research/tree/master/goemotions
2. Emotion Detection Papers: [Will add]
3. HuggingFace Transformers: https://huggingface.co/docs/transformers

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🎓 Academic Information

**Course**: Natural Language Processing  
**Institution**: [Your University]  
**Semester**: [Current Semester]  
**Submission Date**: [Date]

---

**Status**: 🚧 In Development

**Last Updated**: November 26, 2025

