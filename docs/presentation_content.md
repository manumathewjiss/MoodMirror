# MoodMirror: Emotion Classification Project
## Presentation Content & Speaker Notes

---

## Slide 1: Title Slide

**Title:** MoodMirror: Emotion Classification Using Transformer Models

**Subtitle:** A Comparative Study of Baseline and Deep Learning Approaches

**Authors:** [Your Name] & Darshana

**Institution:** [Your Institution]

**Date:** [Insert Date]

---

## Slide 2: Problem Statement & Motivation

### Content:
- **Understanding emotions in text** is crucial for various applications
  - Social media sentiment analysis
  - Mental health monitoring
  - Customer service automation
  - Human-computer interaction

- **Challenge:** Accurately classifying emotions from short, informal text
  - Ambiguity in emotional expression
  - Context-dependent interpretations
  - Class imbalance in real-world data

- **Goal:** Develop an accurate emotion classification system using state-of-the-art NLP techniques

### Speaker Notes:
Introduce the motivation behind emotion classification. Explain why this is an important problem in NLP and what real-world applications it enables. Set the stage for the technical approach.

---

## Slide 3: Dataset Overview

### Content:
- **GoEmotions Dataset** (Google Research)
  - Widely recognized benchmark for emotion classification
  - Originally contains **27 emotion classes**
  - Total samples: **[Insert Total Dataset Size]** (e.g., ~54,000 samples)
  - Split: Training (~80%), Validation (~10%), Test (~10%)

- **Dataset Characteristics:**
  - Source: Reddit comments
  - Text length: Average **[Insert Average Length]** characters, **[Insert Average Words]** words
  - Format: Short, informal social media text

### Speaker Notes:
Explain why you chose the GoEmotions dataset—it's a well-established benchmark that allows for fair comparison with other research. Mention the dataset size and basic statistics. Note that the original dataset has 27 classes, but you'll be working with a reduced set.

**Placeholder:** [Insert Dataset Numbers Here - Total samples, train/val/test splits, average text length]

---

## Slide 4: Emotion Class Selection

### Content:
- **Initial Exploration:** Analyzed all 27 original emotion classes
- **Decision:** Reduced to **7 emotion classes** for better focus and interpretability
  - Anger
  - Disgust
  - Fear
  - Joy
  - Neutral
  - Sadness
  - Surprise

- **Rationale:**
  - Covers primary emotional spectrum
  - Reduces complexity while maintaining meaningful distinctions
  - Aligns with Ekman's basic emotions theory

### Speaker Notes:
Explain the process of selecting 7 classes from the original 27. This was a collaborative decision with your teammate. Mention that you explored different combinations and settled on these 7 as they represent the core emotional categories while being manageable for classification.

---

## Slide 5: Data Preprocessing Pipeline

### Content:
**Preprocessing Steps:**

1. **Text Normalization**
   - Lowercasing all text
   - Removing URLs and web links
   - Removing HTML tags

2. **Special Character Handling**
   - Removing emojis and special Unicode characters
   - Cleaning Reddit-specific patterns (e.g., `[NAME]` placeholders, subreddit references)
   - Normalizing punctuation (removing repeated punctuation marks)

3. **Data Cleaning**
   - Removing very short texts (< 3 characters)
   - Removing email addresses and mentions
   - Preserving essential punctuation for emotion cues

4. **Quality Control**
   - Removed **[Insert Number]** samples with insufficient content
   - Final processed dataset: **[Insert Final Dataset Size]** samples

### Speaker Notes:
Walk through each preprocessing step and explain why it's necessary. Emphasize that preprocessing is crucial for transformer models, though they're more robust than traditional models. Mention that you preserved important punctuation that might carry emotional signals.

**Placeholder:** [Insert Preprocessing Details - Number of removed samples, final dataset size]

---

## Slide 6: Class Imbalance Analysis

### Content:
- **Severe Class Imbalance Detected**
  - Imbalance ratio: **40:1** (most common to least common class)
  - Distribution:
    - Neutral: **[Insert Percentage]**% (dominant class)
    - Joy: **[Insert Percentage]**%
    - Anger: **[Insert Percentage]**%
    - Sadness: **[Insert Percentage]**%
    - Surprise: **[Insert Percentage]**%
    - Fear: **[Insert Percentage]**%
    - Disgust: **[Insert Percentage]**% (minority class)

- **Impact:** Models tend to predict majority class, leading to poor performance on minority classes

- **Solution Strategy:** Implemented class balancing techniques for baseline model

### Speaker Notes:
Explain why class imbalance is a problem—models will be biased toward predicting the majority class. Show the distribution chart if available. Mention that this imbalance ratio of 40:1 is quite severe and requires special handling.

**Placeholder:** [Insert Class Distribution Chart/Table]

---

## Slide 7: Baseline Model - Logistic Regression

### Content:
**Approach:**
- **Model:** Logistic Regression with TF-IDF features
- **Rationale:** Simple, interpretable baseline to establish performance floor

**Class Balancing:**
- Applied **oversampling** for minority classes (below mean)
- Applied **undersampling** for majority classes (above mean)
- Balanced to approximately **[Insert Balanced Class Size]** samples per class
- Total balanced training set: **[Insert Balanced Total]** samples

**Training Configuration:**
- TF-IDF vectorization with appropriate parameters
- Standard train/validation/test split maintained

### Speaker Notes:
Explain why Logistic Regression was chosen as a baseline—it's a simple, interpretable model that provides a performance floor. Detail the balancing strategy: you calculated the mean number of samples per class and balanced all classes to that mean using a combination of oversampling and undersampling.

**Placeholder:** [Insert Balanced Class Size and Total Samples]

---

## Slide 8: Baseline Model Results

### Content:
**Logistic Regression Performance (7-Class):**

- **Accuracy:** **[Insert Accuracy]** (e.g., 53.66%)
- **Precision (Weighted):** **[Insert Precision]** (e.g., 61.79%)
- **Recall (Weighted):** **[Insert Recall]** (e.g., 53.66%)
- **F1-Score (Weighted):** **[Insert F1]** (e.g., 55.56%)

**Analysis:**
- Performance was **not satisfactory** for production use
- Model struggled with minority classes (disgust, fear)
- Better performance on majority classes (neutral, joy)
- Indicates need for more sophisticated approaches

### Speaker Notes:
Present the baseline results honestly. Explain that while the model learned some patterns, the performance wasn't sufficient for a real-world application. This motivates the move to transformer-based models. Mention specific challenges with minority classes.

**Placeholder:** [Insert Baseline Results - Accuracy, Precision, Recall, F1]

---

## Slide 9: Transition to Deep Learning

### Content:
**Why Transformer Models?**
- **Limitations of Baseline:**
  - TF-IDF loses contextual information
  - Cannot capture semantic relationships
  - Limited ability to understand word order and context

- **Transformer Advantages:**
  - Contextualized word embeddings
  - Pre-trained on large corpora
  - Transfer learning capabilities
  - Better handling of informal text

**Models Selected:**
1. **DistilBERT** - Lightweight, faster training
2. **RoBERTa** - Robust training strategy, strong performance

### Speaker Notes:
Explain the transition from traditional ML to deep learning. DistilBERT is a distilled version of BERT that's faster to train while maintaining good performance. RoBERTa uses a more robust training strategy than BERT and often performs better on downstream tasks.

---

## Slide 10: DistilBERT Model

### Content:
**Model Architecture:**
- **Base Model:** DistilBERT-base-uncased
- **Architecture:** 6 transformer layers (vs. 12 in BERT)
- **Parameters:** ~66M (vs. ~110M in BERT)
- **Advantage:** ~60% faster training with minimal performance loss

**Training Configuration:**
- Fine-tuned on preprocessed emotion dataset
- Tokenization: WordPiece tokenizer
- Max sequence length: 512 tokens
- Training with appropriate learning rate and batch size

### Speaker Notes:
Explain DistilBERT's architecture and why it's a good choice—it's faster to train while maintaining competitive performance. Mention the training hyperparameters briefly.

---

## Slide 11: DistilBERT Results

### Content:
**DistilBERT Performance (7-Class):**

- **Accuracy:** **[Insert Accuracy]** (e.g., 63.06%)
- **Precision (Weighted):** **[Insert Precision]** (e.g., 67.42%)
- **Recall (Weighted):** **[Insert Recall]** (e.g., 63.06%)
- **F1-Score (Weighted):** **[Insert F1]** (e.g., 64.06%)

**Improvement over Baseline:**
- **+9.4 percentage points** in accuracy
- **+8.5 percentage points** in F1-score
- Better handling of all emotion classes
- Still room for improvement

### Speaker Notes:
Show the improvement over the baseline. DistilBERT significantly outperforms Logistic Regression, demonstrating the power of transformer models. However, note that performance is still not ideal, which leads to trying RoBERTa.

**Placeholder:** [Insert DistilBERT Results - All Metrics]

---

## Slide 12: RoBERTa Model

### Content:
**Model Architecture:**
- **Base Model:** RoBERTa-base
- **Architecture:** 12 transformer layers, 125M parameters
- **Training Strategy:**
  - Dynamic masking (vs. static in BERT)
  - Larger batch sizes
  - No Next Sentence Prediction task
  - Trained on more data

**Why RoBERTa?**
- Stronger pre-training strategy
- Proven performance on classification tasks
- Better handling of context and semantics

**Training Configuration:**
- Fine-tuned on emotion classification task
- Similar preprocessing and tokenization as DistilBERT

### Speaker Notes:
Explain RoBERTa's improvements over BERT. The dynamic masking and removal of Next Sentence Prediction make it more robust. Mention that RoBERTa is often considered one of the best-performing transformer models for classification tasks.

---

## Slide 13: RoBERTa Results (7-Class)

### Content:
**RoBERTa Performance (7-Class):**

- **Accuracy:** **[Insert Accuracy]** (e.g., 57.94%)
- **Precision (Weighted):** **[Insert Precision]** (e.g., 68.29%)
- **Recall (Weighted):** **[Insert Recall]** (e.g., 57.94%)
- **F1-Score (Weighted):** **[Insert F1]** (e.g., 59.68%)

**Analysis:**
- Higher precision than DistilBERT
- Lower accuracy and recall than DistilBERT
- Performance still **not satisfactory** for 7-class classification
- Indicates task difficulty with fine-grained emotions

**Decision:** Explore reducing number of classes

### Speaker Notes:
Present RoBERTa's results honestly. Interestingly, RoBERTa didn't outperform DistilBERT on all metrics, which might be due to the specific task or hyperparameters. This led to the decision to try a 3-class version, which is a common approach when fine-grained classification is too difficult.

**Placeholder:** [Insert RoBERTa 7-Class Results]

---

## Slide 14: Reducing to 3 Classes

### Content:
**Motivation:**
- 7-class performance was insufficient across all models
- Fine-grained emotion distinctions are challenging
- Many applications benefit from broader categories

**Class Mapping:**
- **Positive:** Joy, Surprise
- **Negative:** Anger, Disgust, Fear, Sadness
- **Neutral:** Neutral

**New Dataset:**
- Reprocessed original data with 3-class mapping
- Maintained same preprocessing pipeline
- Rebalanced dataset for training

### Speaker Notes:
Explain the rationale for reducing to 3 classes. This is a pragmatic decision—sometimes broader categories are more useful and easier to classify accurately. The mapping groups similar emotions together (all negative emotions, all positive emotions).

---

## Slide 15: RoBERTa Results (3-Class)

### Content:
**RoBERTa Performance (3-Class):**

- **Accuracy:** **[Insert Accuracy]** (e.g., 74.64%)
- **Precision (Weighted):** **[Insert Precision]** (e.g., 74.41%)
- **Recall (Weighted):** **[Insert Recall]** (e.g., 74.64%)
- **F1-Score (Weighted):** **[Insert F1]** (e.g., 74.39%)

**Analysis:**
- **Significant improvement** over 7-class version
- **+16.7 percentage points** in accuracy
- Balanced performance across all metrics
- **Satisfactory performance** for practical applications

**Conclusion:** 3-class model selected as final model

### Speaker Notes:
Emphasize the dramatic improvement. The 3-class version achieves much better performance, making it suitable for real-world deployment. This demonstrates the trade-off between granularity and accuracy.

**Placeholder:** [Insert RoBERTa 3-Class Results]

---

## Slide 16: Model Comparison

### Content:
**Comprehensive Model Comparison:**

| Model | Classes | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Logistic Regression | 7 | **[Insert]** | **[Insert]** | **[Insert]** | **[Insert]** |
| DistilBERT | 7 | **[Insert]** | **[Insert]** | **[Insert]** | **[Insert]** |
| RoBERTa | 7 | **[Insert]** | **[Insert]** | **[Insert]** | **[Insert]** |
| RoBERTa | 3 | **[Insert]** | **[Insert]** | **[Insert]** | **[Insert]** |

**Key Findings:**
- Transformer models significantly outperform baseline
- 3-class classification achieves best performance
- RoBERTa-3Class is the optimal model for deployment

### Speaker Notes:
Present the comprehensive comparison. Highlight the progression from baseline to transformers, and the improvement from 7-class to 3-class. This slide should include a visualization if available.

**Placeholder:** [Insert Model Comparison Table/Chart]

---

## Slide 17: Model Comparison Visualization

### Content:
**[Insert Comparison Chart/Figure]**

**Visualization shows:**
- Performance metrics across all models
- Clear improvement trajectory
- Best-performing model highlighted

### Speaker Notes:
If you have a comparison chart, present it here. Walk through the visual, pointing out the improvements at each stage.

**Placeholder:** [Insert Model Comparison Image/Chart]

---

## Slide 18: Deployment - MoodMirror UI

### Content:
**Application Development:**
- **Framework:** Streamlit (Python web framework)
- **Model:** RoBERTa 3-class emotion classifier
- **Features:**
  - Real-time text input and prediction
  - Emotion classification with confidence scores
  - Support for multiple text inputs (up to 4)
  - Emotion trend visualization across inputs
  - User-friendly interface

**Architecture:**
- Backend: Model inference pipeline
- Frontend: Interactive web interface
- Deployment: Localhost demonstration

### Speaker Notes:
Introduce the deployment aspect. Explain that you built a user-friendly interface to demonstrate the model's capabilities. Mention that this is a proof-of-concept that can be extended to a full application.

---

## Slide 19: UI/UX Overview

### Content:
**MoodMirror Interface Features:**

1. **Text Input Section**
   - Single text area for initial input
   - Additional inputs for batch processing (up to 4 texts)

2. **Prediction Display**
   - Color-coded emotion labels (Positive/Neutral/Negative)
   - Confidence scores for predictions
   - Text preview for each prediction

3. **Visualization**
   - Emotion trend chart across multiple inputs
   - Summary statistics (count of each emotion)

4. **User Experience**
   - Clean, modern interface
   - Real-time predictions
   - Reset functionality

### Speaker Notes:
Describe the UI features. Emphasize the user-friendly design and the ability to analyze emotion trends across multiple inputs, which could be useful for analyzing conversation threads or social media posts.

**Placeholder:** [Insert UI Screenshots - Main interface, prediction display, visualization]

---

## Slide 20: UI Screenshots

### Content:
**[Insert UI Screenshot 1: Main Interface]**

**Caption:** MoodMirror main interface with text input

---

**[Insert UI Screenshot 2: Prediction Results]**

**Caption:** Emotion predictions with color-coded labels

---

**[Insert UI Screenshot 3: Trend Visualization]**

**Caption:** Emotion trend analysis across multiple inputs

### Speaker Notes:
Walk through each screenshot, explaining the functionality. Show how users can input text, see predictions, and visualize trends.

**Placeholder:** [Insert UI Screenshots]

---

## Slide 21: Project Summary

### Content:
**What We Accomplished:**

1. ✅ Explored and preprocessed GoEmotions dataset
2. ✅ Established baseline with Logistic Regression
3. ✅ Trained and evaluated DistilBERT model
4. ✅ Trained and evaluated RoBERTa model (7-class)
5. ✅ Developed improved 3-class classification system
6. ✅ Achieved **74.64% accuracy** with RoBERTa-3Class
7. ✅ Built and deployed MoodMirror demonstration UI

**Key Insights:**
- Transformer models significantly outperform traditional ML
- Class granularity vs. accuracy trade-off
- 3-class classification provides optimal balance

### Speaker Notes:
Summarize the entire project journey. Highlight the key achievements and the main insight about the trade-off between granularity and accuracy.

---

## Slide 22: Future Work & Extensions

### Content:
**Planned Extensions:**

1. **Model Improvements**
   - Experiment with larger models (RoBERTa-large)
   - Try ensemble methods
   - Hyperparameter optimization

2. **Application Development**
   - Extend to full end-to-end application
   - Add user authentication and data persistence
   - Implement batch processing for large datasets
   - Add API endpoints for integration

3. **Feature Enhancements**
   - Multi-language support
   - Emotion intensity scoring
   - Context-aware predictions (conversation threads)
   - Real-time streaming analysis

4. **Deployment**
   - Cloud deployment (AWS, GCP, Azure)
   - Mobile application
   - Browser extension for social media analysis

### Speaker Notes:
Present the future roadmap. This shows that this is a study proposal with clear plans for extension. Mention that you plan to develop this into a complete application.

---

## Slide 23: Conclusion

### Content:
**Takeaways:**

- Successfully developed emotion classification system using transformer models
- Demonstrated clear progression from baseline to state-of-the-art approaches
- Achieved satisfactory performance (74.64% accuracy) with 3-class model
- Built functional demonstration UI (MoodMirror)

**Impact:**
- Foundation for emotion-aware applications
- Demonstrates practical application of transformer models
- Provides framework for future extensions

**Thank You!**

### Speaker Notes:
Conclude with a strong summary. Thank the audience and invite questions. Be prepared to discuss technical details, design decisions, and future plans.

---

## Slide 24: Q&A

### Content:
**Questions?**

**Contact Information:**
- [Your Email]
- [Project Repository/GitHub Link]

### Speaker Notes:
Be ready to answer questions about:
- Technical implementation details
- Why certain design decisions were made
- Challenges faced during development
- Comparison with other emotion classification approaches
- Deployment considerations

---

## Additional Notes for Presenter:

### Technical Details to Have Ready:
- Exact dataset sizes (train/val/test splits)
- Preprocessing statistics (samples removed, etc.)
- Training hyperparameters (learning rate, batch size, epochs)
- Training time and computational resources used
- Confusion matrices for each model
- Per-class performance metrics

### Potential Questions & Answers:
1. **Why not use BERT instead of RoBERTa?**
   - RoBERTa's training improvements (dynamic masking, no NSP) generally lead to better performance

2. **Why did RoBERTa perform worse than DistilBERT on 7-class?**
   - Possible reasons: hyperparameters, class imbalance effects, or task-specific characteristics

3. **How did you handle class imbalance in transformer models?**
   - For transformers, we relied on the models' robustness, but class weights could be added

4. **What about other emotion classification datasets?**
   - GoEmotions is widely used, but other datasets (EmoBank, ISEAR) could be explored

5. **How would you improve performance further?**
   - Larger models, ensemble methods, data augmentation, active learning

---

## Presentation Flow Summary:

1. **Introduction** (Slides 1-2): Problem and motivation
2. **Data** (Slides 3-6): Dataset, preprocessing, challenges
3. **Baseline** (Slides 7-8): Logistic Regression approach and results
4. **Deep Learning** (Slides 9-13): DistilBERT and RoBERTa (7-class)
5. **Improvement** (Slides 14-15): 3-class approach and results
6. **Comparison** (Slides 16-17): Comprehensive model comparison
7. **Deployment** (Slides 18-20): MoodMirror UI development
8. **Conclusion** (Slides 21-24): Summary and future work

---

**End of Presentation Content**

