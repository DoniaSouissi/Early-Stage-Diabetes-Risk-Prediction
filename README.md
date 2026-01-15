# 🩺 Early Stage Diabetes Risk Prediction

### **Can Symptoms Alone Predict Diabetes Before a Blood Test?**
This project analyzes a clinical dataset to predict the likelihood of **Early Stage Diabetes**. The goal was to determine if non-invasive symptoms (Polyuria, Age, Gender, Sudden Weight Loss) could be used as an effective pre-screening tool before clinical blood work.

**Key Finding:**
We discovered that **Polyuria (Excess Urination)** and **Polydipsia (Excess Thirst)** are the dominant factors, mathematically outweighing age or genetics in this dataset. The final model achieves **~97% Recall**, ensuring that potential cases are almost never missed—a critical requirement for medical diagnostics.

---

## 📊 Project Results: Model Comparison

We evaluated four different architectures using a 60/20/20 split. To ensure the model generalizes well to new patients, we monitored the gap between **Training Accuracy** and **Validation Accuracy**.

| Model | Train Acc. | Val Acc. | Val Recall | Insight |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 89.1% | 85.6% | 84.4% | Reliable baseline, but struggles with non-linear symptom patterns. |
| **SVM** | 88.8% | 84.6% | 84.4% | Lowest performance; hyperspace margins didn't fit categorical data well. |
| **KNN** | 93.9% | 91.3% | 90.6% | **High performance**; confirms that Age-scaling via StandardScaler was effective. |
| **Random Forest** | **99.7%** | **98.1%** | **96.9%** | **Champion Model:** Near-perfect recall with a very low generalization gap. |

---

## 🛡️ Robustness & Overfitting Check

A key part of our pipeline was ensuring the model didn't just "memorize" the training data:

1.  **Generalization Gap:** The difference between Random Forest's Training (99.7%) and Validation (98.1%) accuracy is less than 2%. This indicates a highly stable model.
2.  **Recall Priority:** In medical screening, a **False Negative** (missing a sick patient) is more dangerous than a **False Positive**. We selected Random Forest because it achieved the highest **Recall (96.9%)**.
3.  **Data Leakage Prevention:** We utilized a strict pipeline where the `StandardScaler` was fit only on training data, ensuring no information from the test set influenced the training process.

## 🔍 The "Gender Bias" & Super-Features

During the evaluation, we uncovered two critical patterns in the data:

1.  **The "Super-Features":**
    * Patients presenting with both **Polyuria** and **Polydipsia** had an overwhelmingly high probability (>90%) of testing positive, regardless of other factors.

2.  **The Gender Imbalance:**
    * **Observation:** A female patient with *zero* symptoms receives a higher base risk score (~55%) than a male with zero symptoms (~45%).
    * **Cause:** This is a reflection of the training data, where a significantly higher percentage of females were diabetic compared to males. The model correctly learned that, statistically, gender is a risk factor in this specific cohort.

---

## 🛠️ Installation & Usage

To replicate this analysis or run the web application:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DoniaSouissi/Early-Stage-Diabetes-Risk-Prediction
    cd diabetes-risk-prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App (Streamlit):**
    Launch the interactive dashboard to test predictions in real-time.
    ```bash
    streamlit run app.py
    ```

4.  **Run the Analysis (Optional):**
    If you want to retrain the models, run the notebooks in order:
    * `notebooks/01_EDA.ipynb`: Discovery of Polyuria/Polydipsia dominance.
    * `notebooks/02_data_preparation.ipynb`: Encoding and Scaling pipeline.
    * `notebooks/03_modeling.ipynb`: Training Random Forest vs. others.
    * `notebooks/04_evaluation.ipynb`: Detailed "What-If" testing.

---

## 📂 Project Structure

```text
├── data/
│   ├── processed/             # Transformed X_train, X_val, X_test etc.
│   └── diabetes_data_upload.csv # Original dataset
├── models/                    # Trained model binaries (.joblib)
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_data_preparation.ipynb  # Cleaning, encoding and Scaling
│   ├── 03_modeling.ipynb          # Model Training & Selection
│   └── 04_evaluation.ipynb        # Performance Metrics & Bias Check
├── utils/
│   ├── preprocessing.py   # Preprocessing functions
│   └── visualization.py   # Plotting helpers
├── .gitattributes                    
├── .gitignore
├── app.py                          # Streamlit Web Application
├── requirements.txt                # Python Dependencies
└── README.md                       # Project Documentation