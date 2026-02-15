# ML Classification App - Assignment 2

## a. Problem Statement
The objective of this assignment is to implement, evaluate, and deploy multiple machine learning classification models to predict whether a breast mass is benign or malignant. The final models are showcased through an interactive Streamlit web application deployed on Streamlit Community Cloud.

## b. Dataset Description
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset.
- **Source**: UCI Machine Learning Repository / Kaggle.
- **Features**: 30 numeric, predictive attributes (e.g., radius, texture, perimeter, area, smoothness). This meets the minimum requirement of 12 features.
- **Instances**: 569 instances. This meets the minimum instance requirement of 500.
- **Target**: Binary classification (Malignant / Benign).

## c. Models Used
Below is the Comparison Table with the evaluation metrics calculated for all 6 implemented models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9596 | 0.9946 | 0.9610 | 0.9292 | 0.9448 | 0.9133 |
| Decision Tree | 0.9895 | 0.9887 | 0.9858 | 0.9858 | 0.9858 | 0.9774 |
| kNN | 0.9438 | 0.9898 | 0.9737 | 0.8726 | 0.9204 | 0.8803 |
| Naive Bayes | 0.9438 | 0.9897 | 0.9545 | 0.8915 | 0.9220 | 0.8793 |
| Random Forest (Ensemble)| 0.9930 | 0.9995 | 0.9952 | 0.9858 | 0.9905 | 0.9850 |
| XGBoost (Ensemble) | 0.9912 | 0.9987 | 0.9905 | 0.9858 | 0.9882 | 0.9812 |

## d. Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Served as a very strong linear baseline, achieving a high AUC, though it had a slightly lower recall compared to its precision. |
| Decision Tree | Showed surprisingly high and balanced performance (0.9858 across Precision, Recall, and F1) for a single tree, indicating highly discriminative features. |
| KNN | Achieved good precision but had the lowest recall (0.8726) among all models, suggesting it struggled slightly to identify all true positive cases. |
| Naive Bayes | Performed similarly to KNN in accuracy but offered a slightly better balance of recall, proving that its feature independence assumption holds reasonably well here. |
| Random Forest (Ensemble) | The best overall performer, achieving the highest Accuracy (0.9930) and a near-perfect AUC (0.9995). It excellently mitigated overfitting while maintaining top-tier precision. |
| XGBoost (Ensemble) | Delivered exceptional, robust performance that nearly matched Random Forest, showing extremely high precision (0.9905) and making very few false positive errors. |
