# ============================================================
# WEEK 4 - AI PROJECT DEPLOYMENT
# Model Training Script
# ============================================================

import os
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

print("Loading Iris dataset...")

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset loaded successfully.")
print("Number of samples:", len(X))
print("Number of features:", X.shape[1])


# ------------------------------------------------------------
# 2. SPLIT DATA
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 3. CREATE MACHINE LEARNING PIPELINE
# ------------------------------------------------------------

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# ------------------------------------------------------------
# 4. TRAIN MODEL
# ------------------------------------------------------------

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ------------------------------------------------------------
# 5. TEST MODEL
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ------------------------------------------------------------
# 6. CREATE MODEL DIRECTORY
# ------------------------------------------------------------

os.makedirs(
    "model",
    exist_ok=True
)


# ------------------------------------------------------------
# 7. SAVE MODEL
# ------------------------------------------------------------

model_path = "model/iris_model.joblib"

joblib.dump(
    {
        "model": model,
        "target_names": list(iris.target_names),   # safe for both list and array
        "feature_names": list(iris.feature_names)  # safe for both list and array
    },
    model_path
)

print("\nModel saved successfully!")

print(
    "Saved to:",
    model_path
)

print("\nTraining process completed.")