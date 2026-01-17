# 🩺 Early Stage Diabetes Risk Prediction

### **Can Symptoms Alone Predict Diabetes Before a Blood Test?**
This project analyzes a clinical dataset to predict the likelihood of **Early Stage Diabetes**. The goal was to determine if non-invasive symptoms (Polyuria, Age, Sudden Weight Loss) could be used as an effective pre-screening tool before clinical blood work.

**Key Finding:**
We discovered that **Polyuria (Excess Urination)** and **Polydipsia (Excess Thirst)** are the dominant factors, mathematically outweighing age or genetics in this dataset. The final model utilizes a **Tuned Random Forest** architecture to achieve **~94% Recall**, ensuring that potential cases are rarely missed—a critical requirement for medical diagnostics.

---

## 📊 Project Results: Model Comparison

We evaluated four different architectures. **Note:** These results are based on a strictly cleaned dataset where **duplicates were removed** to prevent data leakage. This resulted in a smaller, harder, but more realistic testing set.

| Model | Train Acc. | Val Acc. | Precision | Recall | F1-Score | Insight |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **100%** | **92.0%** | **94.3%** | **94.3%** | **94.3%** | **Champion Model:** Best balance of precision and recall. Maintains high performance even without duplicates. |
| **Logistic Regression** | 90.0% | 84.0% | 100% | 77.1% | 87.1% | Excellent Precision (0 False Positives) but misses too many actual cases (Low Recall). |
| **SVM** | 90.7% | 76.0% | 89.7% | 74.3% | 81.3% | Struggled to generalize on the smaller, unique dataset. |
| **KNN** | 93.3% | 72.0% | 95.7% | 62.9% | 75.9% | **Significant Drop:** Performance suffered heavily after removing duplicates, proving it was relying on memorization in earlier tests. |

---

## ⚙️ Hyperparameter Tuning & Stability Check

After selecting Random Forest as the champion model, we performed **Grid Search Optimization (GridSearchCV)** with **5-Fold Cross-Validation** to rigorously test its stability.

* **Objective:** Optimize parameters (Tree Depth, Split Criteria) to maximize **Recall**.
* **Result:** The Tuned Model achieved a Cross-Validation Recall of **94.14%**.
* **The "Stability" Finding:**
    * The cross-validation score (94.14%) was statistically identical to our initial single-split validation score (94.28%).
    * **Conclusion:** This minimal variance proves the model is **Robust**. It performs consistently across different subsets of patients and is not overfitting to a specific "lucky" train-test split. We deployed the stable model with standard parameters (`n_estimators=100`, `max_depth=None`).

---

## 🛡️ Data Integrity & "The Accuracy Drop"

A critical step in our pipeline was the decision to **remove duplicate rows** found during EDA.

1.  **The Discovery:** We found that ~50% of the original dataset consisted of duplicate entries.
2.  **The Decision:** We removed these duplicates to prevent **Data Leakage** (where the model sees the exact same patient in both Train and Test sets).
3.  **The Impact:**
    * Dataset size was reduced from 520 to ~250 rows.
    * **KNN accuracy dropped** from ~91% (with duplicates) to 72% (without).
    * **Random Forest** remained robust (92% Accuracy), proving it learned the actual symptom patterns rather than just memorizing rows.

## 🔍 Bias Mitigation Strategy

During our analysis, we uncovered two critical patterns that influenced our preprocessing:

1.  **The "Super-Features":**
    * Patients presenting with both **Polyuria** and **Polydipsia** had an overwhelmingly high probability of testing positive.

2.  **Gender Bias Removal:**
    * **Observation:** The raw data showed a strong correlation where Female patients were disproportionately diabetic compared to Males, likely due to sampling bias in the hospital data.
    * **Action:** We **removed the Gender column entirely** from the training data. This ensures the model predicts diabetes based solely on **clinical symptoms** (like Thirst, Weight Loss, Polyuria) rather than demographic profiling.

---

## 💻 DiabRisk AI Pro: The Interface

To make the model accessible, we developed **DiabRisk AI Pro**, a fully interactive web application powered by **Streamlit**. Unlike standard data forms, this interface is designed with a **Futuristic Medical Theme** to engage users.

**Key Interface Features:**
* **🧠 Dynamic UX:** Features a "Neural Network" animated background with glassmorphism effects and floating medical particles.
* **⚡ Smart Grouping:** Instead of a long list, symptoms are logically grouped into **Metabolic**, **Neurological**, and **Dermatological** columns for easier data entry.
* **🚫 Bias-Free Design:** The interface strictly implements our research findings by **excluding Gender** from the input fields.
* **📊 Real-Time Feedback:** Provides instant **"Critical Risk"** (Red) or **"System Stable"** (Green) alerts with precise probability percentages.

---

## 🛠️ Installation & Usage

To replicate this analysis or run the web application:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/DoniaSouissi/Early-Stage-Diabetes-Risk-Prediction](https://github.com/DoniaSouissi/Early-Stage-Diabetes-Risk-Prediction)
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
    * `notebooks/01_EDA.ipynb`: Discovery of Polyuria/Polydipsia dominance & Duplicate Handling.
    * `notebooks/02_data_preparation.ipynb`: Encoding, Scaling, and Gender Removal.
    * `notebooks/03_modeling.ipynb`: Training, **Hyperparameter Tuning**, and Model Selection.
    * `notebooks/04_evaluation.ipynb`: Detailed "What-If" testing.

---

## 📂 Project Structure

```text
├── data/
│   ├── processed/             # Cleaned data (No duplicates, No Gender)
│   └── diabetes_data_upload.csv # Original dataset
├── models/                    # Trained model binaries (.joblib)
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis & Integrity Check
│   ├── 02_data_preparation.ipynb  # Cleaning, encoding and Scaling
│   ├── 03_modeling.ipynb          # Model Training, Tuning & Selection
│   └── 04_evaluation.ipynb        # Performance Metrics & Bias Check
├── utils/
│   ├── preprocessing.py   # Preprocessing functions
│   └── visualization.py   # Plotting helpers
├── .gitattributes                    
├── .gitignore
├── app.py                          # Streamlit Web Application
├── requirements.txt                # Python Dependencies
└── README.md                       # Project Documentation