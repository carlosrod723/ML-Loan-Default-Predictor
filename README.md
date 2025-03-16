# Loan Default Prediction Project

## Project Overview

This project focuses on predicting loan defaults using advanced machine learning models. By integrating data from multiple sources and applying state-of-the-art models such as Random Forest and XGBoost, the project aims to provide a robust solution for identifying borrowers who are likely to default on their loans.

## Aim

- **Risk Mitigation:** Improve the identification of potential defaulters to reduce financial risk for lending institutions.
- **Model Performance:** Enhance predictive performance by tuning hyperparameters and comparing multiple models.


## Project Structure

├── LICENSE
├── README.md
├── data
│   ├── Credit_Risk_Dataset.xlsx
│   └── processed_cleaned_data.csv
├── notebooks
│   └── exploration.ipynb
├── requirements.txt
└── src
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-310.pyc
    │   ├── explainability.cpython-310.pyc
    │   ├── model_training.cpython-310.pyc
    │   └── model_utils.cpython-310.pyc
    ├── model_training.py
    └── model_utils.py

## Dataset Used

The dataset used in this project is derived from multiple Excel sheets containing information about loans, employment, personal details, and other relevant financial data. The merged and preprocessed dataset consists of **28,745** records and includes the following columns:

| **Column Name**        | **Definition**                                                                                              |
|------------------------|-------------------------------------------------------------------------------------------------------------|
| **User_id**            | Unique identifier for each loan record.                                                                    |
| **Loan Category**      | Category of the loan (e.g., Consolidation, Credit Card, Home, etc.).                                         |
| **Amount**             | Principal amount of the loan.                                                                               |
| **Interest Rate**      | Interest rate applied to the loan.                                                                          |
| **Tenure(years)**      | Duration of the loan in years.                                                                              |
| **Employmet type**     | Type of employment (e.g., Salaried, Self-employed, etc.).                                                   |
| **Tier of Employment** | Employment tier or rating (e.g., B, D, etc.).                                                                 |
| **Industry**           | Industry of employment (note: values are encrypted and not directly interpretable).                           |
| **Role**               | The role or job title of the borrower.                                                                      |
| **Work Experience**    | Categorical representation of work experience (e.g., 1-2 years, 5-10 years, etc.).                           |
| **Total Income(PA)**   | Total income per annum of the borrower.                                                                     |
| **Gender**             | Gender of the borrower.                                                                                       |
| **Married**            | Marital status of the borrower.                                                                               |
| **Dependents**         | Number of dependents.                                                                                         |
| **Home**               | Housing status (e.g., rent, own, mortgage).                                                                   |
| **Pincode**            | Postal code associated with the borrower's address.                                                         |
| **Social Profile**     | Indicator of the completeness of the borrower's social profile.                                               |
| **Is_verified**        | Verification status of the borrower's information.                                                          |
| **Delinq_2yrs**        | Number of delinquencies in the past 2 years.                                                                  |
| **Total Payement**     | Total amount paid back by the borrower. (Derived feature)                                                     |
| **Received Principal** | Amount of the principal that has been received. (Derived feature)                                             |
| **Interest Received**  | Interest portion received on the loan. (Derived feature)                                                      |
| **Number of loans**    | Total number of loans taken by the borrower.                                                                  |
| **Defaulter**          | Target variable: 1 indicates the borrower defaulted; 0 indicates no default.                                  |

*Note:* Some columns (such as "Industry", "Work Experience", and "Role") were dropped during preprocessing for modeling due to inconsistencies or encryption, but they are documented here for completeness.

## Models Used and Results

### Baseline Models
- **Baseline Random Forest:**  
  - **Accuracy:** 94.93%
  - **Defaulter Recall:** 49%
  - **F1-Score for Defaulters:** 64%
  - **Key Issue:** The baseline model had high overall accuracy but struggled with identifying defaulters, which is critical for loan prediction.

### Tuned Models
- **Tuned Random Forest:**  
  After hyperparameter tuning, the Random Forest model achieved:
  - **Accuracy:** 97.78%
  - **Defaulter Recall:** 80%
  - **F1-Score for Defaulters:** 87%
  - **Implication:** A significant reduction in false negatives, meaning the model now captures a much larger portion of at-risk loans.

- **Tuned XGBoost:**  
  The tuned XGBoost model further improved performance:
  - **Accuracy:** 98.18%
  - **Defaulter Recall:** 83%
  - **F1-Score for Defaulters:** 89%
  - **Implication:** This model exhibits the best performance, particularly in detecting defaulters, making it highly effective for risk management.

## Interpretation for Loan Prediction

- **High Overall Accuracy:**  
  Both tuned models show very high overall accuracy, but accuracy alone can be misleading in imbalanced classification tasks.
  
- **Improved Recall for Defaulters:**  
  The tuned models significantly increase the recall for defaulters. In financial risk management, missing a defaulter (a false negative) can lead to significant losses, so a high recall is crucial.
  
- **Model Comparison:**  
  While both tuned models perform well, the tuned XGBoost model edges out the Random Forest model with a slightly higher recall and F1-score, suggesting that XGBoost may be more effective at capturing the nuances in the data that predict default.
  
- **Business Impact:**  
  Better detection of defaulters means that lenders can more accurately assess risk, potentially leading to better decision-making, reduced losses, and more targeted interventions for at-risk borrowers.

## Conclusion

This project demonstrates how hyperparameter tuning and advanced modeling can improve loan default prediction. By comparing multiple models and understanding their performance through metrics such as recall and F1-score, we are better equipped to identify borrowers at risk of default. The insights gained here provide a strong foundation for further exploration, such as integrating explainable AI techniques to understand model decisions at a granular level.

---

Feel free to modify or expand upon this README as needed. Let me know if you need any further adjustments or additional sections!