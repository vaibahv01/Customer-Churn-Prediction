# 📊 Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer Churn Prediction is a machine learning project that predicts whether a telecom customer is likely to leave a company. The project uses the IBM Telco Customer Churn dataset to analyze customer behavior and identify patterns associated with customer churn.

The objective is to help businesses identify customers at risk of leaving and take proactive steps to improve customer retention.

## 🎯 Project Objectives

- Analyze customer data and identify churn patterns.
- Perform data cleaning and exploratory data analysis (EDA).
- Build and compare multiple machine learning models.
- Select the best model based on evaluation metrics.
- Develop an interactive web application for churn prediction.

## 📂 Dataset

**Dataset:** IBM Telco Customer Churn

- Total records: 7,043 customers
- Original features: 21 columns
- Target variable: Churn
- Churn categories: Yes / No

## ⚙️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## 🔍 Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Train-Test Split
6. Model Building
7. Model Evaluation and Comparison
8. Final Model Selection
9. Model Saving using Joblib
10. Streamlit Web Application Development

## 🤖 Machine Learning Algorithms

The following algorithms were implemented and evaluated:

- Logistic Regression
- Random Forest
- Decision Tree
- Support Vector Machine (SVM)
- Gradient Boosting
- K-Nearest Neighbors (KNN)

### Final Model: Support Vector Machine (SVM)

The SVM model was selected based on its ability to identify customers likely to churn.

**Model Performance:**

| Metric | Score |
|---|---:|
| Accuracy | 74.38% |
| Precision | 51% |
| Recall | 77% |
| F1-Score | 62% |
| ROC-AUC | 82.1% |

## 🚀 Streamlit Application

An interactive web application was developed using Streamlit.

Users can enter customer information, including:
- Customer tenure
- Contract type
- Internet services
- Payment method
- Monthly charges
- Total charges

The application generates:
- Customer churn prediction
- Estimated churn probability
- Basic customer retention suggestions

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/vaibahv01/Customer-Churn-Prediction.git
```

### 2. Navigate to the project folder

```bash
cd Customer-Churn-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── customer_churn_svm_model.pkl
├── requirements.txt
├── README.md
└── Customer_Churn_Analysis.ipynb
```

## 💡 Business Use Case

This project can help telecom companies identify customers who are at risk of leaving their services. By identifying potential churners early, businesses can plan targeted retention campaigns, improve customer experience, and make data-driven decisions.

## 👨‍💻 Author

Developed as an end-to-end Machine Learning and Data Analytics project.

## 📌 Disclaimer

This project is developed for educational and portfolio purposes. Model predictions are estimates and should not be treated as guaranteed customer behavior.
