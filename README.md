# Customer Churn Prediction Pipeline

## Overview
This project implements an **end-to-end Machine Learning pipeline** to predict customer churn using the Scikit-learn Pipeline API. The primary goal is to build a reusable, production-ready workflow that automates data preprocessing and model training while preventing data leakage.

## Dataset
The project utilizes the **Telco Customer Churn Dataset**, which contains information about customers of a telecommunications company and whether they left the service (churned).
- **Source:** [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Target Variable:** `Churn` (Yes/No)

## Requirements
To run this notebook, you need the following Python libraries installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `matplotlib`
- `seaborn`

## Pipeline Architecture
The project leverages Scikit-learn's `Pipeline` and `ColumnTransformer` to handle different data types seamlessly:

1.  **Numerical Features:** (`tenure`, `MonthlyCharges`, `TotalCharges`) are scaled using `StandardScaler`.
2.  **Categorical Features:** (e.g., `gender`, `InternetService`, `Contract`) are encoded using `OneHotEncoder`.
3.  **Model Selection:** The pipeline tests and compares two primary classifiers:
    - **Logistic Regression**
    - **Random Forest Classifier**



## Project Workflow

### 1. Data Cleaning
- Converts `TotalCharges` to a numeric format and handles missing values.
- Maps the target variable `Churn` to binary values (1 for Yes, 0 for No).

### 2. Feature Engineering & Selection
- Separates the dataset into features ($X$) and the target ($y$).
- Splits the data into training (80%) and testing (20%) sets with stratification to maintain class balance.

### 3. Hyperparameter Tuning
Uses `GridSearchCV` to find the optimal parameters for the models:
- **Logistic Regression:** Tunes the regularization strength (`C`).
- **Random Forest:** Tunes `n_estimators`, `max_depth`, and `min_samples_split`.

### 4. Evaluation
Models are evaluated using several metrics:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix** (to visualize True/False Positives and Negatives)
- **ROC Curve & AUC Score**

### 5. Model Deployment Readiness
The entire pipeline—including the preprocessor and the trained model—is saved as a single `churn_model_pipeline.pkl` file using `joblib`. This allows for easy deployment in a production environment without needing to re-run preprocessing steps manually on new data.

## Key Takeaways
- **Clean Workflow:** Pipelines ensure that training and transformation steps are bundled together.
- **No Data Leakage:** Preprocessing parameters are learned only from the training data.
- **Production-Grade:** The saved object is ready to be loaded by a web API or backend service for real-time predictions.