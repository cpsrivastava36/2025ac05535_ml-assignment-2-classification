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
**(https://github.com/cpsrivastava36/2025ac05535_ml-assignment-2-classification)** 

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
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9373 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9561 | 0.9944 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
### Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a strong linear baseline. Standardization is used before classification, which is important because the WDBC features have different scales. Logistic Regression achieved the best overall performance, with the highest Accuracy (98.25%), AUC (99.54%), Precision (98.61%), F1 Score (98.61%) and MCC (96.23%). This indicates that the model provides excellent discrimination and balanced classification performance on the WDBC dataset. |
| Decision Tree | Captures non-linear relationships and is easy to interpret. The tree is constrained with `max_depth=5` and `min_samples_leaf=3` to reduce overfitting. Decision Tree achieved the lowest overall performance among the five models, with an accuracy of 90.35% and MCC of 79.69%. Although it can capture non-linear relationships, its performance was lower than the other models for this dataset.|
| kNN | Performs well after feature standardization because distance-based methods are sensitive to feature scale. Its performance depends on the selected neighborhood size (`k=7`). KNN achieved very strong performance, with 97.37% accuracy and 98.84% AUC. It achieved the highest recall of 100%, meaning all positive-class instances in the test set were correctly identified. Its high performance is supported by feature standardization, which is important for a distance-based algorithm.|
| Naive Bayes | Provides a fast probabilistic baseline. Gaussian Naive Bayes assumes conditional independence and approximately Gaussian feature distributions, so its results can differ from the tree/ensemble methods. Gaussian Naive Bayes achieved 93.86% accuracy and 98.78% AUC. Although its accuracy was lower than Logistic Regression and KNN, its AUC was still high, indicating strong ability to distinguish between the two classes.|
| Random Forest (Ensemble) | Combines many decision trees and generally provides robust non-linear classification performance while reducing the variance of an individual tree. Random Forest achieved 95.61% accuracy and 99.44% AUC. It substantially outperformed the individual Decision Tree, demonstrating the benefit of combining multiple trees through an ensemble approach.|

### Overall Winner
Based on the highest **F1 score** on the held-out test set, the current experiment's winner is:

**Logistic Regression**

Logistic Regression was selected as the overall winner because it achieved the highest accuracy, AUC, precision, F1 score and MCC among the evaluated models. Its recall was also very high at 98.61%. Therefore, it provided the most balanced overall performance on the selected WDBC dataset.

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
    ├── ML_Assignment_2.ipynb
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
