# Telecom Churn Prediction – Machine Learning Project

This project predicts whether a telecom customer is likely to churn (leave the service provider) based on usage patterns, billing information and customer demographics. The objective is to help telecom companies identify customers at risk and improve retention strategies.

---

## Project Overview
A supervised machine learning model was developed using a structured telecom customer dataset. After preprocessing and model evaluation, the final trained model achieved an accuracy of **76.72%**.

The model is integrated into a user-friendly application (Streamlit / GUI) for real-time churn prediction.

---

## Features of the System
- Churn prediction based on customer attributes
- Data preprocessing and feature engineering
- Machine learning model training, validation and evaluation
- Exported model files for real-time prediction
- User interface for entering customer details and viewing prediction results

---

## Machine Learning Workflow

| Phase | Tasks Performed |
|-------|-----------------|
| Data Cleaning | Handling missing values, removing duplicates |
| Preprocessing | Encoding categorical variables, feature scaling |
| Feature Selection | Selecting impactful variables |
| Model Training | Supervised learning with classification algorithms |
| Evaluation | Accuracy, confusion matrix and classification report |
| Deployment | Saved model for real-time prediction |

---

## Dataset Attributes (General)
The dataset includes customer characteristics such as:
- Customer tenure
- Monthly and total charges
- Contract type
- Internet service
- Tech support
- Gender and senior citizen flag
- Payment method and billing preferences
- Churn label (target)

---

## Model Used
The churn prediction system was trained using classification models.  
The final model achieving **76.72% accuracy** was selected for deployment.

Model files typically generated:
```
churn_model.pkl
scaler.pkl
feature_columns.pkl
```

---

## File Structure (Example)

```
Telecom_Churn_Prediction/
│ data/
│ ├─ telecom_churn.csv
│
│ notebooks/
│ ├─ data_processing.ipynb
│ ├─ model_training.ipynb
│
│ src/
│ ├─ churn_model.pkl
│ ├─ scaler.pkl
│ ├─ feature_columns.pkl
│ ├─ predict.py
│
│ app/
│ ├─ main.py  (GUI / Streamlit application)
│ ├─ requirements.txt
│
│ README.md
```

---

## How to Run the Application

1. Install dependencies
```
pip install -r requirements.txt
```

2. Run the prediction app
```
python main.py
```
or for Streamlit:
```
streamlit run main.py
```

3. Enter customer details in the interface and view churn prediction results.

---

## Key Outcome
The project demonstrates how machine learning can support customer retention by identifying high-risk telecom users before they churn. With **76.72% accuracy**, the model provides actionable insights for business intelligence and targeted marketing efforts.

---

## Author
Developer: Brijesh Maurya  
Final Year IT Engineering Student  
Domain Interests: Data Science, Machine Learning, Business Intelligence

