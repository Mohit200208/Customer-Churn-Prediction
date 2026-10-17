# Customer Churn Prediction

An end-to-end machine learning project that predicts
whether a telecom customer is likely to leave a service.

## Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib

## Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Logistic Regression and Random Forest
- Model evaluation and comparison
- Interactive churn prediction web app

## Structure
customer-churn-prediction/
├── data/
├── notebooks/
├── models/
├── app.py
├── requirements.txt
└── README.md

## How to Run

1. Install dependencies:

   pip install -r requirements.txt

2. Place the dataset in the data folder.

3. Train the models:

   python3 src_train.py

4. Start the web app:

   streamlit run app.py

## Evaluation
Models are evaluated using accuracy, precision,
recall, F1-score, and ROC-AUC.

Actual results are generated during training.