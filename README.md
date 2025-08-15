## 📦 Model Download

Due to GitHub's 100 MB file size limit, the trained model file is hosted on Google Drive.

🔗 **[Download Trained Model](https://drive.google.com/drive/folders/1mM40NK5xgqS5SYN7T_aDr-Z-r9Jjbpby?usp=sharing)**  

# imdb-movie-reviews-sentiment-prediction
## 🛠 Tools, Libraries, and Frameworks Used

### **Programming Language**
- **Python 3.x**

### **Data Processing & Analysis**
- **pandas** – for data manipulation and preprocessing  
- **numpy** – for numerical computations  
- **re** & **string** – for text cleaning  
- **BeautifulSoup** – for HTML parsing  
- **nltk** – for stopwords, tokenization, and synonym replacement  

### **Machine Learning & Deep Learning**
- **scikit-learn** – for TF-IDF vectorization, label encoding, cross-validation, and evaluation metrics  
- **tensorflow / keras** – for building and training Bi-LSTM + Bi-GRU models  
- **joblib** – for saving/loading tokenizers and models  

### **Feature Extraction**
- **TF-IDF (Term Frequency–Inverse Document Frequency)** – for transforming text into numerical feature vectors  

### **Pre-trained Embeddings**
- **GloVe (Global Vectors for Word Representation)**  

### **Data Augmentation & Translation**
- **wordnet** (from NLTK) – for synonym replacement  
- **googletrans** – for back translation  

### **Visualization**
- **matplotlib** – for plotting training/validation metrics and graphs  
- **seaborn** – for heatmaps and statistical visualization  

### **Environment / Platforms**
- **Kaggle** – for model training and experimentation
- **Google Drive / gdown** – for dataset downloading and model storage

## Machine Learning Approach: IMDB Movie Reviews Classification

A complete workflow for **text classification** using TF-IDF features and multiple machine learning classifiers.

### Key Steps

- **Data Loading & Inspection**
  - Load training, test, and extra test datasets
  - Explore data structure and sample reviews

- **Text Preprocessing**
  - Remove HTML tags, punctuation, numbers
  - Convert to lowercase and remove stopwords
  - Store cleaned text in a new column

- **Feature Extraction**
  - Convert text to TF-IDF features
  - Limit vocabulary to top 42,000–43,000 words

- **Train/Test Split**
  - 80/20 split with stratification
  - Separate features (X) and labels (y)

- **Model Training & Evaluation**
  - Train multiple classifiers:  
    Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest, Gradient Boosting  
  - Evaluate with Accuracy, Precision, Recall, F1-score, and confusion matrices
  - Perform 5-fold stratified cross-validation for robust evaluation

- **Hyperparameter Optimization**
  - Use GridSearchCV with 5-fold cross-validation

- **Stacking Ensemble**
  - Combine base classifiers with Logistic Regression as final estimator
  - Evaluate ensemble with cross-validation and test sets

- **Model Saving**
  - Save trained classifiers and TF-IDF vectorizer as `.pkl` files for future use

- **Demo / Inference**
  - Interactive sentiment prediction using saved models and vectorizer
  - Allows real-time testing without retraining

## Deep Learning Approach: Hybrid Bi-LSTM + Bi-GRU

A hybrid **Bidirectional LSTM + GRU model** for binary text classification.

- **Preprocessing:** Text cleaning, tokenization, and padding  
- **Architecture:** Embedding layer → Bi-LSTM + Bi-GRU → Dense layers with Dropout  
- **Training:** Binary cross-entropy loss, sigmoid activation, EarlyStopping & ReduceLROnPlateau  
- **Inference:** Predict on test/demo sentences (interactive or notebook mode)  
- **Evaluation:** Confusion matrix, classification report, accuracy/loss curves  
- **Flexibility:** Works in both interactive and non-interactive notebook environments

### Machine Learning Model Insights

1. **Logistic Regression**  
   - **Accuracy:** 0.8791 | **Precision:** 0.8782 | **Recall:** 0.8804 | **F1-Score:** 0.8793  
   - Logistic Regression performed **best overall**, achieving the highest accuracy and a balanced precision-recall. It is a simple yet strong baseline for classification tasks and handles linearly separable data well.

2. **Linear SVM**  
   - **Accuracy:** 0.8690 | **Precision:** 0.8772 | **Recall:** 0.8581 | **F1-Score:** 0.8675  
   - Linear SVM had slightly lower accuracy than Logistic Regression but **high precision**, indicating fewer false positives. Recall is slightly lower, showing it misses some true positives.

3. **Multinomial Naive Bayes (NB)**  
   - **Accuracy:** 0.8340 | **Precision:** 0.8715 | **Recall:** 0.7835 | **F1-Score:** 0.8252  
   - NB achieved **high precision** but lower recall, meaning it predicts positive cases confidently but misses several actual positives. Works well with text or count-based features.

4. **Random Forest**  
   - **Accuracy:** 0.8507 | **Precision:** 0.8591 | **Recall:** 0.8390 | **F1-Score:** 0.8489  
   - Random Forest provided a **good balance** of precision and recall, performing moderately well across all metrics. Its ensemble nature helps reduce overfitting.

5. **Gradient Boosting**  
   - **Accuracy:** 0.8298 | **Precision:** 0.8060 | **Recall:** 0.8688 | **F1-Score:** 0.8362  
   - Gradient Boosting had the **highest recall** among ML models but lower precision, meaning it identifies most positives but also produces more false positives.

6. **Stacking Ensemble**  
   - **Accuracy:** 0.8772 | **Precision:** 0.8893 | **Recall:** 0.8615 | **F1-Score:** 0.8752  
   - Stacking Ensemble combined multiple models to achieve **high precision and competitive accuracy**, making it one of the strongest performers among ML approaches. It benefits from leveraging strengths of individual classifiers.

### Deep Learning Model Insight

1. **Hybrid Bi-GRU + Bi-LSTM**  
   - **Accuracy:** 0.8327 | **Precision:** 0.8382 | **Recall:** 0.8246 | **AUC:** 0.9133  
   - This model combines **Bidirectional GRU and LSTM layers** to capture sequential patterns in the data.  
   - It achieved a **high AUC**, indicating excellent ranking ability and strong discrimination between classes.  
   - While the accuracy is slightly lower than the top ML models, it provides **better balance in precision and recall** for imbalanced datasets.  
   - Deep learning models like this can capture complex patterns that classical ML models may miss, especially in sequential or text-based data.

# Sentiment Prediction Demo

This script supports **real-time sentiment prediction** using either **ML classifiers** or a **Hybrid Bi-LSTM + Bi-GRU deep learning model**.

## Features

- **Predefined Sentences**
  - Predicts sentiment and probability for demo sentences.
  - ML: Uses vectorizer (TF-IDF/CountVectorizer) + classifier.
  - DL: Uses embedding + Bi-LSTM/Bi-GRU layers.

- **Interactive Mode**
  - Users can type custom sentences.
  - Type `exit` to quit.
  - Shows predicted sentiment and probability.

- **Multiple Classifier/Model Support**
  - ML: Logistic Regression, SVM, Random Forest, Naive Bayes.
  - DL: Hybrid Bi-LSTM + Bi-GRU.
  - Ensemble options (stacking) available for ML.
ChatGPT said:

**The image shows a console output of a sentiment prediction demo. It includes predictions for predefined sentences and an interactive mode where the user inputs custom sentences, with each sentence displaying the predicted sentiment (Positive/Negative) and its probability.**
<img width="955" height="556" alt="demo_script" src="https://github.com/user-attachments/assets/23f1ea88-e661-4dd4-97e0-758ac0ecccb2" />
