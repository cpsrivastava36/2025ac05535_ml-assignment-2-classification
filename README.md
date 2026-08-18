# Machine Learning Assignment 2 — Classification Models

## a. Problem statement
Build and compare multiple classification models on one public classification dataset,
evaluate them using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation
Coefficient (MCC), and deploy an interactive Streamlit application.

## b. Dataset description
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)

The dataset is a binary classification dataset from the UCI Machine Learning Repository
and is available through `sklearn.datasets.load_breast_cancer`.

- Instances: 569
- Input features: 30
- Target: diagnosis
- Class 0: Malignant
- Class 1: Benign
- Train/test split: 80/20
- Stratification: Yes
- Random state: 42

The assignment requires at least 12 features and 500 instances; this dataset satisfies both
requirements.

## c. Github Repository Link
**To be updated after GitHub upload:** `<YOUR_GITHUB_REPOSITORY_LINK>`

## d. Models used

Five models explicitly listed in the assignment were implemented on the same train/test split:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

> Note: The assignment text says “all 6 ML models”, but the model list and comparison-table
> template contain five models. This project follows the five models explicitly listed in the
> assignment.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |\n| Decision Tree | 0.9035 | 0.9373 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |\n| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |\n| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |\n| Random Forest | 0.9561 | 0.9944 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |\n
### Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a strong linear baseline. Standardization is used before classification, which is important because the WDBC features have different scales. |
| Decision Tree | Captures non-linear relationships and is easy to interpret. The tree is constrained with `max_depth=5` and `min_samples_leaf=3` to reduce overfitting. |
| kNN | Performs well after feature standardization because distance-based methods are sensitive to feature scale. Its performance depends on the selected neighborhood size (`k=7`). |
| Naive Bayes | Provides a fast probabilistic baseline. Gaussian Naive Bayes assumes conditional independence and approximately Gaussian feature distributions, so its results can differ from the tree/ensemble methods. |
| Random Forest (Ensemble) | Combines many decision trees and generally provides robust non-linear classification performance while reducing the variance of an individual tree. |

### Overall Winner
Based on the highest **F1 score** on the held-out test set, the current experiment's winner is:

**Logistic Regression**

The exact winner should be discussed using the full metric table rather than accuracy alone.

## Streamlit application
The application provides:
- CSV test-data upload
- Model-selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Prediction output for uploaded test records

## Repository structure

```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_results.csv
├── metadata.json
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deploy `app.py` from the GitHub repository using Streamlit Community Cloud.

**Live Streamlit App Link:** `<YOUR_STREAMLIT_APP_LINK>`

## Academic-integrity note
The project is intentionally structured so the student can inspect, understand, test and
customize the implementation before submission. Do not submit unchanged AI-generated work.
Review the code, create your own GitHub commit history, run the experiment in BITS Virtual
Lab, capture the required screenshot, and replace the placeholder links.
