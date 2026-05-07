# Customer Churn Prediction using Machine Learning

## Overview

This project focuses on predicting customer churn using Machine Learning techniques. Customer churn refers to customers who discontinue a company’s services. Predicting churn helps businesses take preventive actions to improve customer retention and reduce revenue loss.

The project implements a complete end-to-end Machine Learning pipeline including:

* Data preprocessing
* Missing value handling
* Categorical feature encoding
* Class imbalance handling using SMOTE
* Model training and evaluation
* Cross-validation
* Feature importance analysis
* Model serialization using Pickle

The models used in this project are:

* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

Among all models, Random Forest achieved the best performance.

---

# Problem Statement

Customer churn is a major challenge for subscription-based and telecom businesses. Companies lose revenue when customers discontinue services.

The goal of this project is to build a predictive model that can identify customers likely to churn based on their demographic and service-related information.

---

# Dataset

Dataset Used:

Telco Customer Churn Dataset

The dataset contains customer-related information such as:

* Gender
* Senior citizen status
* Contract type
* Internet service
* Monthly charges
* Total charges
* Payment method
* Tenure
* Streaming services
* Technical support
* Churn status

Dataset Shape:

* Rows: 7043
* Columns: 21

Target Variable:

* Churn

  * Yes → Customer left the service
  * No → Customer retained the service

---

# Technologies Used

## Programming Language

* Python

## Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* Pickle

## Development Environment

* Visual Studio Code

---

# Machine Learning Workflow

## 1. Data Loading

The dataset is loaded using Pandas.

## 2. Data Cleaning

* Removed unnecessary columns
* Converted TotalCharges to numeric format
* Handled missing values

## 3. Feature Encoding

Categorical features were converted into numerical format using Label Encoding.

## 4. Handling Imbalanced Dataset

The dataset had more non-churn customers than churn customers.

SMOTE (Synthetic Minority Oversampling Technique) was used to balance the classes.

## 5. Train-Test Split

The dataset was split into:

* 80% Training Data
* 20% Testing Data

## 6. Model Training

Three Machine Learning models were trained:

* Decision Tree
* Random Forest
* XGBoost

## 7. Cross Validation

5-fold cross-validation was performed to evaluate model stability.

## 8. Model Evaluation

The models were evaluated using:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score

## 9. Model Saving

The best-performing model and encoders were saved using Pickle.

---

# Project Results

## Best Model

Random Forest Classifier

## Performance Metrics

* Accuracy: ~77%
* Precision: ~58%
* Recall: ~57%
* F1-Score: ~57%

The Random Forest model performed best due to its ability to reduce overfitting and capture complex patterns in the dataset.

---

# Project Structure

```text
customer_churn_project/
│
├── main.py
├── best_model.pkl
├── encoders.pkl
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md
└── requirements.txt
```

---

# How to Run the Project

## Step 1: Clone Repository

```bash
git clone <repository-link>
```

## Step 2: Open Project Folder

```bash
cd customer_churn_project
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Run the Project

```bash
python main.py
```

---

# Future Improvements

Possible future enhancements include:

* Streamlit web application
* Hyperparameter tuning
* ROC curve visualization
* SHAP explainability
* Flask/FastAPI deployment
* Real-time prediction dashboard

---

# Learning Outcomes

Through this project, the following concepts were implemented and understood:

* Machine Learning classification
* Data preprocessing
* Feature engineering
* Handling imbalanced datasets
* Cross-validation
* Model evaluation metrics
* Ensemble learning
* Model serialization

---

# Conclusion

This project successfully demonstrates the use of Machine Learning techniques for predicting customer churn. The implemented pipeline can help businesses identify customers likely to discontinue services and take proactive retention measures.

Designed and implemented a customer churn prediction system leveraging supervised machine learning techniques to identify customers likely to discontinue services.

Applied feature preprocessing, label encoding, and oversampling techniques (SMOTE) to improve model robustness on imbalanced datasets.

Compared multiple classification models including Decision Tree, Random Forest, and XGBoost using 5-fold cross-validation and performance metrics.

Utilized Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, and Seaborn for data analysis, visualization, and predictive modeling.

The project also highlights the importance of preprocessing, class balancing, and model evaluation in building reliable predictive systems.
