# 🫀 Heart Failure Prediction using Machine Learning

> **College Project — B.Tech CSE-AI&DS**
> **Manas Singh | IIIT Manipur**

---

## 📌 Project Overview

This project predicts the survival of patients suffering from **Congestive Heart Failure (CHF)** using clinical records and machine learning models. It includes full exploratory data analysis (EDA), feature engineering, model comparison, SHAP-based interpretability, and a live Flask web application for real-time predictions.

---

## 🗂️ Project Structure

```
MANAS_heart_failure_clinical_records/
│
├── heart_failure_clinical_records_dataset.csv  # Dataset (299 records, 13 features)
├── CHF_Prediction.ipynb                        # Main Jupyter Notebook (full ML pipeline)
├── manas.ipynb                                 # Exploratory / scratch notebook
│
├── app.py                                      # Flask backend for prediction API
├── index.html                                  # Frontend UI (served by Flask)
│
├── best_model.pkl                              # Saved best ML model (pickle)
├── scaler.pkl                                  # Saved feature scaler (pickle)
│
├── all_roc_curves.png                          # ROC curves for all models
├── best_model_evaluation.png                   # Confusion matrix & metrics of best model
├── boxplots.png                                # Feature distribution boxplots
├── charts1.png                                 # General EDA charts
├── correlation_heatmap.png                     # Feature correlation heatmap
├── knn_analysis.png                            # KNN hyperparameter analysis
├── model_comparison.png                        # Accuracy/F1 comparison across models
├── shap_dot_plot.png                           # SHAP beeswarm / dot plot
├── shap_importance.png                         # SHAP feature importance bar chart
│
├── requirements.txt                            # Python dependencies
└── README.md                                   # Project documentation (this file)
```

---

## 📊 Dataset Description

- **Source:** UCI Machine Learning Repository — Heart Failure Clinical Records Dataset
- **Records:** 299 patients
- **Target Variable:** `DEATH_EVENT` (0 = Survived, 1 = Died)
- **Class Distribution:** 203 survived (67.9%) | 96 died (32.1%)

### Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the patient (years) |
| `anaemia` | Binary | Decrease of red blood cells (0/1) |
| `creatinine_phosphokinase` | Numeric | Level of CPK enzyme in blood (mcg/L) |
| `diabetes` | Binary | Patient has diabetes (0/1) |
| `ejection_fraction` | Numeric | % of blood leaving heart each contraction |
| `high_blood_pressure` | Binary | Patient has hypertension (0/1) |
| `platelets` | Numeric | Platelets in blood (kiloplatelets/mL) |
| `serum_creatinine` | Numeric | Level of creatinine in blood (mg/dL) |
| `serum_sodium` | Numeric | Level of sodium in blood (mEq/L) |
| `sex` | Binary | Female = 0, Male = 1 |
| `smoking` | Binary | Patient smokes (0/1) |
| `time` | Numeric | Follow-up period (days) |
| `DEATH_EVENT` | Binary | **Target** — Patient died during follow-up (0/1) |

---

## ⚙️ ML Pipeline

### 1. Exploratory Data Analysis (EDA)
- Distribution plots, boxplots, and correlation heatmap
- No missing values found in the dataset

### 2. Feature Engineering
Four additional binary features were derived:

| Engineered Feature | Condition |
|---|---|
| `low_ejection` | `ejection_fraction < 40` |
| `is_elderly` | `age > 65` |
| `high_creatinine` | `serum_creatinine > 1.5` |
| `low_sodium` | `serum_sodium < 135` |

### 3. Models Trained & Compared
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Curves
- Confusion Matrix

### 5. Explainability
- **SHAP (SHapley Additive exPlanations)** used to explain model predictions
- Feature importance ranked via SHAP dot plot and bar chart

---

## 🌐 Web Application

A real-time prediction web app built with **Flask**.

### How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure these files are in the same folder:
#    app.py | best_model.pkl | scaler.pkl | index.html

# 3. Start the Flask server
python app.py

# 4. Open in browser
# http://localhost:5000
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the prediction UI |
| `/predict` | POST | Returns prediction + probability (JSON) |
| `/health` | GET | Health check — returns model type |

### Sample `/predict` Request

```json
{
  "age": 75,
  "anaemia": 0,
  "creatinine_phosphokinase": 582,
  "diabetes": 0,
  "ejection_fraction": 20,
  "high_blood_pressure": 1,
  "platelets": 265000,
  "serum_creatinine": 1.9,
  "serum_sodium": 130,
  "sex": 1,
  "smoking": 0,
  "time": 4
}
```

### Sample Response

```json
{
  "prediction": 1,
  "label": "HIGH RISK - May Not Survive",
  "probability": 87.43,
  "status": "success"
}
```

---

## 📦 Requirements

```
flask
numpy
pandas
scikit-learn
shap
matplotlib
seaborn
pickle5
jupyter
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📈 Key Results

- Best model selected based on highest ROC-AUC and F1-Score on test set
- SHAP analysis revealed **ejection fraction**, **serum creatinine**, and **follow-up time** as the most influential predictors of mortality

---

## 👤 Author

| Field | Detail |
|---|---|
| **Name** | Manas Singh |
| **Institute** | Indian Institute of Information Technology (IIIT), Manipur |
| **Department** | Computer Science & Engineering — AI & Data Science (CSE-AD) |
| **Project Type** | College Project |

---

## 📜 License

This project is for academic and educational purposes only.
Dataset credit: [UCI ML Repository — Heart Failure Clinical Records](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)
