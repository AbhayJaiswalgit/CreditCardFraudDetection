# Credit Card Fraud Detection

## 🔍 Project Overview
This project aims to detect **fraudulent credit card transactions** using machine learning. It covers both **supervised** and **unsupervised models**, applies **resampling techniques** (undersampling, SMOTE), and adjusts probability thresholds to optimize fraud detection while minimizing false positives.

The project is implemented in a single **Jupyter notebook**.

## 🧰 Tools & Libraries
- Python 3
- `pandas`, `numpy`
- `scikit-learn` (Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting)
- `xgboost` (XGBClassifier)
- `imblearn` (SMOTE)
- `matplotlib`, `seaborn` (visualizations)
- `joblib` (saving/loading models)

---

## ⚙️ Methodology

1. **Data Preprocessing**
   - Missing value handling
   - Feature scaling
   - Train-test split

2. **Handling Imbalance**
   - Random undersampling
   - SMOTE oversampling

3. **Model Training**
   - Supervised: Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost
   - Unsupervised: Isolation Forest

4. **Threshold Tuning**
   - Adjusted probability thresholds to optimize fraud detection metrics (precision, recall, ROC-AUC).

5. **Evaluation Metrics**
   - ROC-AUC
   - Precision, Recall, F1-Score
   - Confusion Matrix

---

## 📥 Dataset Download

The dataset used is the **Credit Card Fraud Detection dataset** from Kaggle. To download:

1. Go to the Kaggle dataset page: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click **Download**.
3. Extract the CSV file into the same folder as the notebook or adjust the notebook path accordingly.

> Note: You need a Kaggle account to download the dataset.

---

## 📊 Key Results

### Supervised Models (SMOTE + Threshold)

| Model | Threshold | Test ROC-AUC | Precision | Recall | F1-Score | Observations |
|-------|-----------|---------------|-----------|--------|-----------|--------------|
| XGBoost | 0.71 | 0.977 | 0.30 | 0.85 | 0.44 | Best balance of recall, precision, and generalization |
| Logistic Regression | 0.71 | 0.964 | 0.12 | 0.86 | 0.21 | Lightweight, interpretable backup model |
| Random Forest | 0.50 | 0.961 | 0.89 | 0.74 | 0.80 | High precision, minor overfitting |
| Decision Tree | 0.50 | 0.825 | 0.36 | 0.65 | 0.46 | Overfits easily; weaker generalization |
| Isolation Forest | N/A | 0.941 | 0.07 | 0.78 | 0.12 | Unsupervised baseline; many false positives |

✅ **Conclusion:** XGBoost with SMOTE and threshold 0.71 is the recommended model for detecting fraud effectively.

---

## 💡 Insights
- SMOTE helps tree-based models capture minority (fraud) class better.
- Threshold adjustment improves precision without significantly harming recall.
- Logistic Regression remains a fast, interpretable alternative.
- Isolation Forest is useful as a pre-screening anomaly detector but insufficient for final classification.

---

## 🚀 How to Run
1. Download the dataset from Kaggle as described above.
2. Open `CreditCard_FraudDetection.ipynb` in **Jupyter Notebook** or **Google Colab**.
3. Follow the notebook to:
   - Preprocess the data
   - Train and evaluate models
   - Tune thresholds and compare results

---

## ⚖️ License
This project is for **educational purposes**. Refer to the dataset license on Kaggle.
