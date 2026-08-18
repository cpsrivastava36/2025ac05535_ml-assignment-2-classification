import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WDBC Classification",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

# Restore the original title and add a smaller dataset link below it
st.title("🧬 ML Assignment : Wisconsin Diagnostic Breast Cancer Classification")

st.markdown(
    "##### [UCI WDBC Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)"
)

st.markdown(
    """
    This Streamlit application evaluates five machine learning classification
    models trained on the Wisconsin Diagnostic Breast Cancer (WDBC)
    dataset.
    """
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib"
}


# ============================================================
# REQUIRED FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    loaded_models = {}

    for name, path in MODEL_PATHS.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        loaded_models[name] = joblib.load(path)

    return loaded_models


try:
    models = load_models()

except Exception as e:

    st.error(f"Unable to load models: {e}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose a classification model",
    list(models.keys())
)

selected_model = models[selected_model_name]


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("1. Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Upload the test_data.csv file to evaluate the selected model."
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"Unable to read CSV file: {e}")
    st.stop()


st.success(
    f"Successfully loaded {data.shape[0]} rows "
    f"and {data.shape[1]} columns."
)


# ============================================================
# TARGET
# ============================================================

if "target" not in data.columns:

    st.error(
        "The uploaded file must contain a 'target' column "
        "for evaluation."
    )

    st.stop()


# ============================================================
# FEATURE VALIDATION
# ============================================================

missing_features = [
    feature
    for feature in EXPECTED_FEATURES
    if feature not in data.columns
]

extra_features = [
    column
    for column in data.columns
    if column not in EXPECTED_FEATURES + ["target"]
]


if missing_features:

    st.error("The following required features are missing:")

    st.write(missing_features)

    st.stop()


if extra_features:

    st.warning(
        "Extra columns were found and will be ignored:"
    )

    st.write(extra_features)


# ============================================================
# PREPARE DATA
# ============================================================

X_test = data[EXPECTED_FEATURES]
y_test = data["target"]


# ============================================================
# SHOW DATA
# ============================================================

st.header("2. Uploaded Test Data")

st.dataframe(
    data.head(10),
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

try:

    y_pred = selected_model.predict(X_test)

    y_prob = selected_model.predict_proba(X_test)[:, 1]

except Exception as e:

    st.error(
        f"Prediction failed: {e}"
    )

    st.stop()


# ============================================================
# METRICS
# ============================================================

st.header(
    f"3. Evaluation Metrics — {selected_model_name}"
)


accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ============================================================
# METRIC DISPLAY
# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

col2.metric(
    "AUC",
    f"{auc:.4f}"
)

col3.metric(
    "Precision",
    f"{precision:.4f}"
)


col4, col5, col6 = st.columns(3)

col4.metric(
    "Recall",
    f"{recall:.4f}"
)

col5.metric(
    "F1 Score",
    f"{f1:.4f}"
)

col6.metric(
    "MCC",
    f"{mcc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("4. Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"],
    ax=ax,
    annot_kws={"size": 18, "weight": "bold"},
    cbar_kws={"shrink": 0.5}
)

ax.set_xlabel("Predicted Label", fontsize=14)
ax.set_ylabel("Actual Label", fontsize=14)
ax.set_title(f"Confusion Matrix — {selected_model_name}", fontsize=22)

ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.header("5. Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=["Malignant", "Benign"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ============================================================
# PREDICTIONS
# ============================================================

st.header("6. Benign Probability Report")

class_names = {
    0: "Malignant",
    1: "Benign"
}

prediction_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Benign Probability": y_prob
})

prediction_df["Actual"] = prediction_df["Actual"].map(class_names)
prediction_df["Predicted"] = prediction_df["Predicted"].map(class_names)

# Round the "Benign Probability" column to 4 decimal places for better readability
prediction_df["Benign Probability"] = (
    prediction_df["Benign Probability"].round(4)
)

st.dataframe(
    prediction_df,
    use_container_width=True
)
# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "2025AC05535 Machine Learning Assignment 2 | "
    "Wisconsin Diagnostic Breast Cancer Dataset"
)