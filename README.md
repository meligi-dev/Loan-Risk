# Project Title
Loan Risk

## Introduction
The Loan Risk project aims to evaluate and predict the likelihood of a loan application being approved or denied based on various applicant characteristics. By leveraging historical applicant data, this project builds a robust predictive model that helps financial institutions make data-driven decisions and mitigate default risks.

## Problem Statement
Financial institutions often face the challenge of accurately assessing the risk associated with approving loans. Manual evaluation processes can be slow and prone to human error, leading to potential financial losses or missed opportunities. The objective of this project is to develop a machine learning classifier capable of predicting the `LoanApproved` status (0 for Denied, 1 for Approved) using the applicant's demographic and financial features.

## Dataset Description
The dataset used in this project contains 5000 rows and 10 columns describing various attributes of loan applicants:
- **Age**: The applicant's age.
- **Income**: The applicant's annual income.
- **LoanAmount**: The requested loan amount.
- **CreditScore**: The applicant's credit score.
- **YearsExperience**: Number of years the applicant has been employed.
- **Gender**: The applicant's gender (dropped during preprocessing).
- **Education**: The highest education level attained by the applicant (High School, Bachelor, Master, PhD).
- **City**: The city where the applicant resides.
- **EmploymentType**: The applicant's current employment status (Salaried, Self-Employed, Unemployed).
- **LoanApproved**: The target variable (1 = Approved, 0 = Denied).

## Methodology

### Data Cleaning
- Dropped the `Gender` column as it was deemed unnecessary for the prediction.
- Addressed missing values in the `Income` and `CreditScore` columns by imputing the median.
- Handled missing values in the `Education` column by applying mode imputation. 

### EDA (Exploratory Data Analysis)
- Created distribution plots to understand the spread of features in the dataset.
- Analyzed feature correlations with the target variable `LoanApproved`. `CreditScore` showed a medium positive correlation (~0.46), and being `Unemployed` showed a moderate negative correlation (~-0.33) with loan approval.
- Visualized class balance, revealing a higher count of denied loans compared to approved loans.

### Modeling
- **Encoding**: Mapped ordinal variables (`Education`) to numerical values. Used One-Hot Encoding (`pd.get_dummies`) for nominal categorical variables (`City`, `EmploymentType`).
- **Data Splitting**: Divided the dataset into training (80%) and testing (20%) sets using stratified splitting to preserve the class proportion.
- **Scaling**: Standardized the features using `StandardScaler`.
- **Handling Imbalance**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to address class imbalance.
- **Algorithms Evaluated**: Trained Logistic Regression, Support Vector Machine (SVM), Random Forest, and XGBoost classifiers.

## Results
The trained models were evaluated on the test set using Accuracy and F1-Score metrics:

- **Logistic Regression**: Accuracy = 86.30%, F1 = 74.77%
- **SVM**: Accuracy = 89.80%, F1 = 79.60%
- **XGBoost**: Accuracy = 95.60%, F1 = 90.18%
- **Random Forest**: Accuracy = 96.20%, F1 = 91.48%

**Best Model**: The **Random Forest** algorithm performed the best overall, yielding the highest accuracy and F1-score on the testing data. The fitted model, scaler, and columns have been exported using `joblib` for prospective deployment.

## Conclusion
The Loan Risk prediction model successfully demonstrates how algorithmic approaches can effectively evaluate an applicant's viability for a loan. `CreditScore` and certain employment status types proved to be highly crucial factors in the approval process. The Random Forest model's strong predictive performance (96.2% Accuracy, 91.5% F1) makes it highly dependable for deployment. 

Future work could include extensive hyperparameter tuning (e.g., using GridSearchCV or RandomizedSearchCV) to push the evaluation metrics further, as well as integrating the model into a web application for real-time risk assessment.
