# 🩺 Early Stage Diabetes Risk Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
This project is a Machine Learning solution designed to predict the likelihood of **Early Stage Diabetes** based on specific symptoms and demographic data. 

Using a dataset from the **UCI Machine Learning Repository**, we analyzed key risk factors (such as Polyuria, Polydipsia, and Gender) and built a predictive model with **~97% Recall**, ensuring that potential cases are rarely missed. The solution includes a full data pipeline, rigorous evaluation, and a user-friendly **Streamlit Web App** for real-time predictions.

---

## 📂 Project Structure
```text
├── data/
│   ├── diabetes_data_upload.csv    # Raw Dataset
│   └── processed/                  # Transformed X_train, X_test, etc.
├── models/
│   ├── best_model.joblib           # Trained Random Forest Model
│   ├── preprocessor.joblib         # ColumnTransformer (Scaling/Encoding)
│   └── target_encoder.joblib       # LabelEncoder for Target
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_data_preparation.ipynb   # Splitting, Scaling, Encoding
│   ├── 03_modeling.ipynb           # Model Training & Comparison
│   └── 04_evaluation.ipynb         # Detailed Testing & ROC Analysis
├── utils/
│   ├── preprocessing.py   # Preprocessing functions
│   └── visualization.py   # Plotting helpers
├── .gitattributes                    
├── .gitignore
├── app.py                          # Streamlit Web Application
├── requirements.txt                # Python Dependencies
└── README.md                       # Project Documentation

