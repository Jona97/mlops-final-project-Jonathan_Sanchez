# 🧠 MLOps Final Project - Insurance Charges Prediction

## 📌 Student Information
- **Full Name:** Jonathan David Sánchez Abanto  
- **E-mail:** jonathan.sanchez.a@uni.pe 
- **Grupo:** 2  

---

# 📊 Project Name:
## 💰 Insurance Medical Charges Prediction using Machine Learning

---

# 📖 1. Problem Definition

Medical insurance companies need to estimate the healthcare expenses of their clients based on personal attributes such as:

- Age
- Sex
- BMI
- Smoking habits
- Region
- Number of children

Incorrect estimation of insurance charges may lead to:
- Financial losses for insurance companies
- Incorrect pricing strategies
- Risk mismanagement

### 🎯 Goal:
Develop a Machine Learning model capable of predicting medical insurance charges for new clients based on demographic and lifestyle information.

---

# 📁 2. Data Acquisition

Dataset used:

📌 Kaggle - Medical Cost Personal Dataset  
https://www.kaggle.com/datasets/mirichoi0218/insurance

The dataset contains:

| Feature    | Description |
|-----------|------------|
| age       | Age of primary beneficiary |
| sex       | Gender |
| bmi       | Body Mass Index |
| children  | Number of dependents |
| smoker    | Smoking status |
| region    | Residential region |
| charges   | Medical insurance cost (target) |

---

# 🔬 3. ML Experimentation

Experiments were conducted using Jupyter Notebooks located in:

notebooks/


### ✔ Exploratory Data Analysis
Notebook:

notebooks/00_eda_insurance.ipynb


Includes:
- Distribution analysis
- Correlation analysis
- Boxplots
- Feature relationships

---

### ✔ Model Training
Notebook:

notebooks/01_train_insurance.ipynb


Models evaluated:
- Linear Regression
- Random Forest Regressor

Metrics used:
- MAE
- RMSE
- R² Score

Experiments tracked using:

✅ MLflow

Results stored in:

reports/train_results.json


---

# 🛠 4. ML Development Activities

## 📌 Data Preparation

Raw Dataset stored in:


data/insurance/insurance.csv


Data preparation logic implemented in:


src/data_preparation.py


Includes:
- Feature separation
- Categorical encoding
- Numerical scaling
- Train/Test split

---

## 📌 Model Training Implementation

Training pipeline implemented in:


src/train.py


Includes:
- Pipeline creation
- Model training
- Cross-validation
- Model evaluation
- MLflow logging

Serialized models saved in:


models/


Artifacts:
- best_model.pkl
- best_model.joblib
- MLflow model directory

Champion model selected based on lowest RMSE.

---

# 🚀 5. Model Deployment & Serving

Model is served locally using a Flask REST API.

Serving logic implemented in:


src/serving.py


Run locally with:

```bash
python -m src.serving

Server runs at:

http://127.0.0.1:5000
🌐 Prediction via Web Interface

A simple HTML form was implemented to manually input client information and generate predictions.

Template located in:

src/templates/predict.html

Access prediction interface at:

http://127.0.0.1:5000/predict

Input example:

{
  "age": 19,
  "sex": "female",
  "bmi": 27.9,
  "children": 0,
  "smoker": "yes",
  "region": "southwest"
}

Output:

Predicted insurance medical charges.

📈 ML Lifecycle Implementation

This project follows the Machine Learning Lifecycle:

Problem Definition

Data Collection

Data Cleaning & Preprocessing

Exploratory Data Analysis

Feature Engineering

Model Training

Model Evaluation

Model Deployment

Model Serving

Lifecycle Diagram:

📦 Requirements Installation

Create virtual environment:

python -m venv .venv

Activate:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
📌 Run Training Pipeline
python -m src.train
📌 Run Serving API
python -m src.serving
✅ Expected Result

Successful prediction of insurance charges via:

REST API

Web Interface

📤 Delivery

This project includes:

✔ Source Code
✔ Dataset
✔ Notebooks
✔ Trained Models
✔ Reports
✔ Flask API
✔ HTML UI
✔ MLflow Experiments

All final components are located in the main branch.
