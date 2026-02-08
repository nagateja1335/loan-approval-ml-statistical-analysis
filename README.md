#  Loan Prediction: Hypothesis Testing & Supervised Modeling

##  Major Project — Machine Learning & Statistical Analysis

This project predicts **loan approval status** by integrating **inferential statistical analysis** with **supervised machine learning models**. It demonstrates how financial institutions can leverage data science techniques to automate loan decision-making and minimize credit risk.

---

##  Project Overview

Loan approval is a critical process in banking and finance. Manual verification of loan applications is time-consuming and prone to bias. This project builds a data-driven system that:

* Validates statistical relationships using hypothesis testing
* Identifies key predictors of loan approval
* Trains machine learning classification models
* Evaluates model performance using standard metrics

---

##  Objectives

* Perform inferential statistical analysis
* Conduct T-Test and Chi-Square tests
* Study feature relationships using correlation
* Preprocess and encode financial data
* Select top predictive features
* Train supervised ML models
* Evaluate and compare model performance

---

##  Dataset Description

The Loan Prediction dataset contains demographic and financial information of loan applicants.

### Key Features

* Gender
* Married
* Dependents
* Education
* Self_Employed
* ApplicantIncome
* LoanAmount
* Credit_History
* Property_Area
* Loan_Status (Target)

Total Records: **614**

---

##  Data Preprocessing

Data preprocessing steps included:

* Removal of `Loan_ID` (identifier column)
* Handling missing values using mode & median imputation
* Label Encoding of categorical variables
* Correlation analysis using heatmap
* Removal of weak features:

  * CoapplicantIncome
  * Loan_Amount_Term

---

##  Correlation Analysis

A correlation heatmap was generated to analyze relationships between features and loan approval.

**Key Insight:**
`Credit_History` showed the strongest correlation with Loan_Status, making it the most influential predictor.

---

##  Hypothesis Testing

###  T-Test

Compared ApplicantIncome between approved and rejected loans.

* T-Statistic: −0.116
* P-Value: 0.907

Result: No significant difference in income between approved and rejected applicants.

---

###  Chi-Square Test

Tested association between Education and Loan_Status.

* Chi-Square: 4.091
* P-Value: 0.043

Result: Significant relationship exists between education and loan approval.

---

###  ANOVA (Conceptual)

ANOVA would be used to compare mean loan amounts across multiple income groups.

---

##  Feature Selection

Feature selection was performed using **SelectKBest (Chi-Square scoring)** to identify the most influential predictors.

Top features included:

* Credit_History
* ApplicantIncome
* LoanAmount
* Education
* Property_Area

---

## 🤖 Machine Learning Models

Two supervised classification models were trained:

### 1️. Logistic Regression

Baseline linear classifier.

### 2️. Decision Tree Classifier

Tree-based nonlinear classifier.

---

##  Model Evaluation

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | 78.86%   | 0.703   |
| Decision Tree       | 69.10%   | 0.633   |

Logistic Regression achieved better performance.

---

##  ROC Curve Analysis

ROC curves were plotted to evaluate classification capability.

Logistic Regression showed a higher AUC, indicating better class separation.

---

##  Key Insights

* Credit_History is the strongest predictor.
* Education significantly impacts approval.
* Income alone does not determine approval.
* Feature selection improved model performance.
* Statistical testing supported ML findings.

---

##  Final Outcome

Logistic Regression was selected as the best model for loan prediction based on:

* Higher Accuracy
* Better ROC-AUC Score
* Balanced classification performance

---

##  Future Scope

* Implement Random Forest & XGBoost
* Apply SMOTE for imbalance handling
* Perform hyperparameter tuning
* Deploy as a web application
* Integrate real-time banking datasets

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn
* Jupyter Notebook / VS Code

---

##  Project Structure

```
loan-prediction-project/
│
├── dataset/
│   └── loan_data.csv
│
├── notebooks/
│   └── loan_prediction.ipynb
│
├── src/
│   └── loan_prediction_major_project.py
│
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License.

---

⭐ If you found this project useful, consider giving it a star!
