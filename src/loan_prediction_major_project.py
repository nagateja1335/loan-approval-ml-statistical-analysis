"""
Major Project
Loan Prediction: Hypothesis Testing & Supervised Modeling

This script performs:

• Data preprocessing
• Correlation analysis
• Feature removal
• Hypothesis testing
• Feature selection
• Model training
• Evaluation
• ROC curve analysis
"""

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

# Load Dataset
print("\nLoading Dataset...\n")

df = pd.read_csv("loan_data.csv")

print(df.head())

# Remove Identifier Column
df.drop("Loan_ID", axis=1, inplace=True)

# Handle Missing Values
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

numerical_cols = df.select_dtypes(include=np.number).columns

for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("\nMissing Values Handled.\n")

# Encode Categorical Features
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("\nEncoding Completed.\n")

# Correlation Heatmap
plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap — Loan Dataset")
plt.show()

# Remove Weak Features
df.drop(
    ["CoapplicantIncome", "Loan_Amount_Term"],
    axis=1,
    inplace=True
)

print("\nWeak Features Removed.\n")

# Hypothesis Testing
print("\n--- Hypothesis Testing ---\n")

# T-Test
approved = df[df['Loan_Status'] == 1]['ApplicantIncome']
rejected = df[df['Loan_Status'] == 0]['ApplicantIncome']

t_stat, p_val_t = stats.ttest_ind(approved, rejected)

print("T-Test Results")
print("T-Statistic:", t_stat)
print("P-Value:", p_val_t)

# Chi-Square Test
cont_table = pd.crosstab(
    df['Education'],
    df['Loan_Status']
)

chi2_stat, p_val_chi, dof, exp = stats.chi2_contingency(cont_table)

print("\nChi-Square Results")
print("Chi-Square:", chi2_stat)
print("P-Value:", p_val_chi)

# Feature Selection
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

selector = SelectKBest(score_func=chi2, k=10)

X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]

print("\nTop Selected Features:\n", selected_features)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# Model Training
# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

# Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

# Evaluation Metrics
print("\n--- Logistic Regression ---\n")

print("Accuracy:",
      accuracy_score(y_test, y_pred_log))

print("Confusion Matrix:\n",
      confusion_matrix(y_test, y_pred_log))

print("ROC-AUC:",
      roc_auc_score(y_test, y_pred_log))

print("\n--- Decision Tree ---\n")

print("Accuracy:",
      accuracy_score(y_test, y_pred_tree))

print("Confusion Matrix:\n",
      confusion_matrix(y_test, y_pred_tree))

print("ROC-AUC:",
      roc_auc_score(y_test, y_pred_tree))

# ROC Curve Analysis
# Logistic Regression ROC
y_prob_log = log_model.predict_proba(X_test)[:,1]

fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)

roc_auc_log = auc(fpr_log, tpr_log)

# Decision Tree ROC
y_prob_tree = tree_model.predict_proba(X_test)[:,1]

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)

roc_auc_tree = auc(fpr_tree, tpr_tree)

# Plot ROC Comparison
plt.figure(figsize=(8,6))

plt.plot(fpr_log, tpr_log,
         label=f"Logistic Regression (AUC = {roc_auc_log:.2f})")

plt.plot(fpr_tree, tpr_tree,
         label=f"Decision Tree (AUC = {roc_auc_tree:.2f})")

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Project End
print("\nProject Execution Completed Successfully!")