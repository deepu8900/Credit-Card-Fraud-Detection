# Credit-Card-Fraud-Detection
# Credit Card Fraud Detection

## Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Fraud detection is critical for financial institutions to prevent monetary losses and protect customers. The goal is to build a model that can accurately classify transactions as fraudulent or legitimate.

## Dataset
The dataset used is the **Credit Card Fraud Detection** dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains transactions made by European cardholders in September 2013. The dataset is highly imbalanced with only a small fraction of transactions labeled as fraud.

- Number of transactions: 284,807
- Number of frauds: 492 (0.172%)
- Features: Numerical inputs from PCA transformation (for privacy reasons)
- Target variable: `Class` (1 for fraud, 0 for non-fraud)

## Data Preprocessing
- Data was checked for missing values and outliers.
- Features were scaled/normalized as needed.
- Due to class imbalance, techniques like undersampling, oversampling, or SMOTE can be applied (optional, depending on your notebook).
  
## Model(s) Used
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Other classifiers can be tested for comparison.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC Curve

Since the dataset is imbalanced, precision, recall, and F1-score are more reliable than accuracy alone.

## Results
Summarize your best performing model results here. For example:

| Model             | Precision | Recall | F1-Score | ROC-AUC |
|-------------------|-----------|--------|----------|---------|
| Random Forest     | 0.95      | 0.85   | 0.90     | 0.98    |
| Logistic Regression| 0.92      | 0.80   | 0.85     | 0.95    |

_Add charts and graphs here if available._

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/deepu8900/Credit-Card-Fraud-Detection.git
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Open the Jupyter Notebook and run the cells:

bash
Copy
Edit
jupyter notebook Credit_Card_Fraud_Detection.ipynb
Dependencies
Python 3.x

pandas

numpy

scikit-learn

matplotlib

seaborn

imbalanced-learn (if used)

Future Work
Experiment with deep learning models like Neural Networks.

Improve data balancing techniques.

Deploy the model as a web service for real-time fraud detection.

References
Kaggle Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud

Imbalanced-learn Documentation

Various machine learning tutorials on fraud detection.

