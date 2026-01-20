💳 Credit Card Fraud Detection (ML Model Comparison)
This project focuses on detecting fraudulent credit card transactions using Machine Learning.
The notebook trains and compares multiple ML models and selects the best one based on performance metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC. 
credit card fraud detection.ipy…


🚀 Features
✅ Loads and preprocesses the dataset (creditcard.csv)
✅ Handles imbalanced data using SMOTE oversampling
✅ Trains and evaluates 5 different ML models
✅ Compares models using metrics + visualization
✅ Saves the best model + scaler + feature columns using joblib 
credit card fraud detection.ipy…


🧠 Models Used
This project compares the following models:

Logistic Regression

Random Forest

XGBoost

LightGBM

CatBoost 
credit card fraud detection.ipy…


📊 Evaluation Metrics
Each model is evaluated using:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Score

Confusion Matrix 
credit card fraud detection.ipy…


⚙️ Libraries & Tools
pandas, numpy

scikit-learn

imblearn (SMOTE)

xgboost, lightgbm, catboost

matplotlib, seaborn

joblib 
credit card fraud detection.ipy…


📂 Dataset
The dataset used is creditcard.csv, which contains anonymized transaction data.
It includes a highly imbalanced target column where fraud cases are rare.

Place the dataset here:

bash
Copy code
/content/creditcard.csv
(or update the path in the notebook) 
credit card fraud detection.ipy…


🏗️ Workflow (Pipeline)
Load Dataset

Preprocess & Scale Features

Train-Test Split

Balance Training Data using SMOTE

Train Models

Evaluate Models

Compare Models with Graphs

Save Best Model 
credit card fraud detection.ipy…


💾 Saved Files (Output)
After training, the best model and required components are saved as:

best_model.pkl

scaler.pkl

feature_cols.pkl 
credit card fraud detection.ipy…


▶️ How to Run
1️⃣ Install Required Packages
bash
Copy code
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost matplotlib seaborn joblib
2️⃣ Run Notebook
Open the notebook in Google Colab / Jupyter Notebook and run all cells.

📌 Results
The notebook prints metrics for each model and shows a comparison graph for ROC-AUC score. 
credit card fraud detection.ipy…


👤 Author
Akash Pandey
CSE Student | AI/ML Learner

