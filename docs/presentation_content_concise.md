# MoodMirror: Emotion Classification Project
## Concise Presentation Content (12 Slides)

---

## Slide 1: Title & Overview

**Title:** MoodMirror: Emotion Classification Using Transformer Models

**Subtitle:** A Comparative Study of Baseline and Deep Learning Approaches

**Authors:** [Your Name] & Darshana | **Date:** [Insert Date]

**Quick Overview:**
- Dataset: GoEmotions (Google Research) - **[Insert: 54,263 samples]**
- Models: Logistic Regression → DistilBERT → RoBERTa
- Best Result: **74.64% accuracy** (RoBERTa, 3-class)
- Deployment: MoodMirror UI (Streamlit)

---

## Slide 2: Problem Statement & Dataset

**Problem:**
- Classify emotions in short, informal text (Reddit comments)
- Applications: Sentiment analysis, mental health monitoring, customer service

**GoEmotions Dataset:**
- **Total:** **[Insert: 54,263 samples]** | Train: **[Insert: 43,410]** | Val: **[Insert: 5,426]** | Test: **[Insert: 5,427]**
- **Source:** Reddit comments | **Avg length:** **[Insert: 68 chars, 12.8 words]**
- **Original:** 27 emotion classes → Reduced to **7 classes** (Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise)
- **Challenge:** Severe class imbalance (**40:1 ratio**)

**Figure:** `emotion_distribution.png` - Emotion class distribution in dataset

---

## Slide 3: Data Preprocessing

**Objective:** Transform raw Reddit comments into clean, tokenization-ready text while preserving emotion signals

**Preprocessing Pipeline:**

1. **Text Normalization:**
   - Convert to lowercase for consistency
   - Remove URLs (http/https/www links)
   - Strip HTML tags

2. **Reddit-Specific Cleaning:**
   - Remove Reddit patterns: `[NAME]`, `r/subreddit` references
   - Remove email addresses and @mentions
   - Handle hashtags (remove #, keep word)

3. **Emoji & Special Characters:**
   - Comprehensive emoji removal (all Unicode emoji ranges)
   - Remove skin tone modifiers and zero-width joiners
   - Clean special characters while preserving punctuation

4. **Punctuation Normalization:**
   - Normalize repeated punctuation (e.g., `!!!` → `!`)
   - Normalize repeated hyphens (e.g., `---` → `-`)
   - **Preserve emotion cues:** `!`, `?`, `.`, `,`, `:`, `;`, quotes, parentheses

5. **Whitespace & Quality Control:**
   - Normalize multiple spaces to single space
   - Remove leading/trailing non-word characters
   - **Filter short texts:** Remove samples < 3 characters
     - Removed: Train=48, Val=4, Test=4 (**Total: 56 samples**)

**Results:**
- **Before:** 54,263 samples
- **After:** 54,207 samples (Train: 43,362 | Val: 5,422 | Test: 5,423)
- **Average text length:** **[Insert: 66 chars, 12.6 words]**
- **Outcome:** Clean, consistent text ready for transformer tokenization

**Figures:**
- `text_length_analysis.png` - Text length distribution (characters and words)
- `top_words.png` - Most frequent words in processed dataset

---

## Slide 4: Baseline Model - Logistic Regression

**Approach:**
- **Model:** Logistic Regression with TF-IDF features
- **Class Balancing:** Oversampling + Undersampling to **[Insert: ~6,194 samples per class]**
- **Total Balanced Training Set:** **[Insert: 43,358 samples]**

**Results (7-Class):**
- **Accuracy:** **[Insert: 53.66%]**
- **Precision:** **[Insert: 61.79%]** | **Recall:** **[Insert: 53.66%]** | **F1:** **[Insert: 55.56%]**

**Analysis:** Performance not satisfactory → Need more sophisticated models

**Figures:** 
- `confusion_matrix_baseline.png` - Confusion matrix
- `baseline_model_performance.png` - Performance metrics visualization

---

## Slide 5: DistilBERT Model

**Why DistilBERT?**
- Lightweight BERT variant (6 layers vs. 12)
- ~60% faster training, competitive performance
- Good for initial transformer baseline

**Training:**
- Base: `distilbert-base-uncased`
- Fine-tuned on balanced 7-class dataset
- Tokenization: WordPiece, max length: 512

**Results (7-Class):**
- **Accuracy:** **[Insert: 63.06%]** | **F1:** **[Insert: 64.06%]**
- **Improvement:** +9.4% accuracy over baseline

**Figure:** `confusion_matrix_distilbert.png` - Confusion matrix for DistilBERT

---

## Slide 6: RoBERTa Model (7-Class)

**Why RoBERTa?**
- Robust training strategy (dynamic masking, no NSP)
- Strong performance on classification tasks
- 12 layers, 125M parameters

**Training:**
- Base: `roberta-base`
- Fine-tuned on 7-class emotion dataset

**Results (7-Class):**
- **Accuracy:** **[Insert: 57.94%]** | **F1:** **[Insert: 59.68%]**
- **Analysis:** Higher precision but lower accuracy than DistilBERT
- **Conclusion:** 7-class classification still challenging

**Figure:** `confusion_matrix_roberta.png` - Confusion matrix for RoBERTa (7-class)

---

## Slide 7: Reducing to 3 Classes

**Motivation:**
- 7-class performance insufficient across all models
- Fine-grained distinctions too challenging
- Broader categories more practical for applications

**Class Mapping:**
- **Positive:** Joy, Surprise
- **Negative:** Anger, Disgust, Fear, Sadness  
- **Neutral:** Neutral

**Approach:**
- Reprocessed dataset with 3-class mapping
- Maintained same preprocessing pipeline
- Rebalanced for training

**Figures:**
- `emotion_distribution_3class.png` - 3-class emotion distribution
- `split_comparison_3class.png` - Distribution across train/val/test splits (3-class)

---

## Slide 8: RoBERTa Results (3-Class)

**RoBERTa Performance (3-Class):**

- **Accuracy:** **[Insert: 74.64%]**
- **Precision:** **[Insert: 74.41%]** | **Recall:** **[Insert: 74.64%]** | **F1:** **[Insert: 74.39%]**

**Key Improvements:**
- **+16.7 percentage points** over 7-class version
- Balanced performance across all metrics
- **Satisfactory for practical deployment**

**Conclusion:** Selected as final model

**Figure:** `confusion_matrix_roberta_3class.png` - Confusion matrix for RoBERTa (3-class)

---

## Slide 9: Model Comparison

**Comprehensive Results:**

| Model | Classes | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Logistic Regression | 7 | **[Insert: 53.66%]** | **[Insert: 61.79%]** | **[Insert: 53.66%]** | **[Insert: 55.56%]** |
| DistilBERT | 7 | **[Insert: 63.06%]** | **[Insert: 67.42%]** | **[Insert: 63.06%]** | **[Insert: 64.06%]** |
| RoBERTa | 7 | **[Insert: 57.94%]** | **[Insert: 68.29%]** | **[Insert: 57.94%]** | **[Insert: 59.68%]** |
| **RoBERTa** | **3** | **[Insert: 74.64%]** | **[Insert: 74.41%]** | **[Insert: 74.64%]** | **[Insert: 74.39%]** |

**Key Findings:**
- Transformers significantly outperform baseline (+9-21% accuracy)
- 3-class achieves optimal balance (granularity vs. accuracy)
- RoBERTa-3Class selected for deployment

**Figures:**
- `model_comparison.png` - Detailed model comparison visualization
- `model_comparison_combined.png` - Combined comparison chart
- `Confusion Matrix for all models.png` - All confusion matrices side-by-side

---

## Slide 10: Deployment - MoodMirror UI

**Application:**
- **Framework:** Streamlit (Python web app)
- **Model:** RoBERTa 3-class emotion classifier
- **Features:**
  - Real-time text input and prediction
  - Color-coded emotion labels (Positive/Neutral/Negative)
  - Batch processing (up to 4 texts)
  - Emotion trend visualization
  - Confidence scores

**Architecture:**
- Backend: Model inference pipeline
- Frontend: Interactive web interface
- Deployment: Localhost demonstration

**Note:** UI screenshots to be captured from running Streamlit app (`src/deployment/emotion_app.py`)

---

## Slide 11: Project Summary & Future Work

**Educational Value: Comprehensive NLP Learning Journey**

This project served as a **hands-on study** of end-to-end NLP pipeline development, covering:

**1. Data Acquisition & Exploration:**
- Working with real-world datasets (GoEmotions from Google Research)
- Understanding dataset characteristics, class distribution, and data quality
- Learning to make informed decisions about class reduction (27 → 7 → 3 classes)

**2. Data Preprocessing Mastery:**
- Text normalization techniques (lowercasing, URL/HTML removal)
- Domain-specific cleaning (Reddit patterns, emoji handling)
- Quality control and filtering strategies
- Understanding the impact of preprocessing on model performance

**3. Class Imbalance Handling:**
- Identifying severe imbalance (40:1 ratio)
- Implementing oversampling and undersampling techniques
- Learning the critical importance of balanced datasets for classification

**4. Baseline Model Development:**
- Traditional ML approach (Logistic Regression with TF-IDF)
- Establishing performance benchmarks
- Understanding feature engineering for text classification

**5. Transformer Architecture Deep Dive:**
- **DistilBERT:** Lightweight transformer, faster training, knowledge distillation
- **RoBERTa:** Advanced BERT variant with improved training strategies
- Fine-tuning pre-trained models for downstream tasks
- Learning transfer learning in NLP

**6. Model Evaluation & Comparison:**
- Understanding accuracy, F1-score, confusion matrices
- Analyzing model performance across different class granularities
- Making data-driven decisions about model selection

**7. Deployment & Application Development:**
- Building interactive UI with Streamlit
- Model inference pipeline development
- Creating user-friendly interfaces for ML models

**Project Achievements:**
✅ Processed GoEmotions dataset (54K+ samples)  
✅ Established baseline (Logistic Regression: 53.66% accuracy)  
✅ Trained DistilBERT (63.06% accuracy, 7-class)  
✅ Trained RoBERTa (57.94% accuracy, 7-class)  
✅ Developed 3-class system (74.64% accuracy)  
✅ Built MoodMirror demonstration UI  

**Key Learning Insights:**
- Transformer models outperform traditional ML by 9-21%
- Class granularity vs. accuracy trade-off in real-world applications
- Importance of iterative refinement based on model performance
- End-to-end pipeline development from data to deployment

**Future Work: Transforming Study Project into Full Application**

**Vision:** Develop a **full-fledged emotional tracking application for students**

**Technical Enhancements:**
- **Model Improvements:** Larger models (RoBERTa-large), ensemble methods, hyperparameter optimization, active learning
- **Feature Expansion:** Multi-language support, context-aware emotion detection, temporal emotion tracking
- **Performance:** Model compression for mobile deployment, real-time inference optimization

**Application Development:**
- **Student Dashboard:** Track emotional patterns over time, identify stress periods, mood trends
- **Analytics & Insights:** Weekly/monthly emotional reports, pattern recognition, personalized recommendations
- **Integration:** Calendar integration, study session tracking, correlation with academic performance
- **Privacy & Ethics:** Secure data handling, student privacy protection, ethical AI considerations

**Deployment Roadmap:**
- **Phase 1:** Cloud deployment (AWS/GCP), API endpoints for scalability
- **Phase 2:** Mobile application (iOS/Android) for on-the-go tracking
- **Phase 3:** Browser extension for seamless integration with study tools
- **Phase 4:** Institutional deployment with admin dashboards

**Impact:** This study project provides the foundation for building emotion-aware educational tools that can support student well-being and academic success.

---

## Slide 12: Conclusion

**Project Summary:**
- Successfully developed end-to-end emotion classification system
- Achieved 74.64% accuracy with RoBERTa-3Class
- Demonstrated practical application through MoodMirror UI
- Comprehensive learning experience in modern NLP techniques

**Takeaways:**
- Real-world NLP projects require iterative refinement
- Understanding the problem domain (emotion classification) is crucial
- Balance between model complexity and practical usability
- Foundation established for future emotional tracking applications

**Thank You! Questions?**

---

## Speaker Notes Summary:

### Slide 1: Quick intro - set the stage
### Slide 2: Explain dataset choice and class reduction rationale
### Slide 3: Brief preprocessing overview - emphasize importance
### Slide 4: Baseline establishes performance floor
### Slide 5: First transformer - shows improvement
### Slide 6: More powerful model but still struggling with 7 classes
### Slide 7: Pivot strategy - explain trade-off
### Slide 8: Success! Highlight the improvement
### Slide 9: Comprehensive comparison - show progression
### Slide 10: Practical demonstration
### Slide 11: Summarize journey and insights
### Slide 12: Future vision and wrap-up

---

## Total: 12 Slides

**Structure:**
1. Title (1)
2. Problem & Dataset (1)
3. Preprocessing (1)
4. Baseline (1)
5. DistilBERT (1)
6. RoBERTa 7-class (1)
7. 3-class Approach (1)
8. RoBERTa 3-class Results (1)
9. Comparison (1)
10. Deployment (1)
11. Summary (1)
12. Future Work & Conclusion (1)

All essential information included in concise format.

---

## Available Figures Reference:

**Dataset & Exploration:**
- `emotion_distribution.png` - 7-class emotion distribution (Slide 2)
- `emotion_distribution_3class.png` - 3-class emotion distribution
- `text_length_analysis.png` - Text length statistics (7-class)
- `text_length_analysis_3class.png` - Text length statistics (3-class)
- `split_comparison.png` - Distribution across train/val/test splits (7-class)
- `split_comparison_3class.png` - Distribution across train/val/test splits (3-class)
- `top_words.png` - Most frequent words in dataset

**Model Results:**
- `baseline_model_performance.png` - Baseline metrics visualization (Slide 4)
- `confusion_matrix_baseline.png` - Baseline confusion matrix (Slide 4)
- `confusion_matrix_distilbert.png` - DistilBERT confusion matrix (Slide 5)
- `confusion_matrix_roberta.png` - RoBERTa 7-class confusion matrix (Slide 6)
- `confusion_matrix_roberta_3class.png` - RoBERTa 3-class confusion matrix (Slide 8)
- `model_comparison.png` - Model comparison visualization (Slide 9)
- `model_comparison_combined.png` - Combined comparison chart (Slide 9)
- `Confusion Matrix for all models.png` - All confusion matrices combined (Slide 9)

**All figures located in:** `results/figures/`

