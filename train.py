"""
COM572 – Machine Learning Coursework: Task 1
Project Title: Predicting Sleep Disorders Using Machine Learning

Training Script: train.py
Author: [Student Name]
Date: April 2026

Description:
    This script implements a complete, reproducible machine learning pipeline
    to predict sleep disorders (None, Insomnia, Sleep Apnoea) from the
    Sleep Health and Lifestyle dataset. It trains and evaluates three models
    (Logistic Regression, Decision Tree, Random Forest), performs
    cross-validation, and applies hyperparameter tuning to the final model.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

# =============================================================================
# 2. DATA LOADING & INSPECTION
# =============================================================================

def load_and_inspect(filepath: str) -> pd.DataFrame:
    """
    Load the dataset, inspect its structure, and return a copy.
    Missing values in 'Sleep Disorder' represent individuals with no disorder.
    """
    df = pd.read_csv(filepath)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape       : {df.shape}")
    print(f"Columns     : {list(df.columns)}")
    print(f"\nClass distribution (raw):\n{df['Sleep Disorder'].value_counts(dropna=False)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print("=" * 60)

    return df

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features:
      - Drop the surrogate key 'Person ID'
      - Fill NaN in 'Sleep Disorder' → 'None' (no disorder)
      - Parse 'Blood Pressure' into two numeric features (Systolic, Diastolic)
      - Standardise column names (strip whitespace)
      - Encode BMI Category (ordinal encoding preserves clinical order)
    """
    df = df.copy()

    # --- Drop surrogate key (no predictive value) ---
    df.drop(columns=["Person ID"], inplace=True)

    # --- Target: NaN → 'None' (no sleep disorder) ---
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

    # --- Parse Blood Pressure string '126/83' → systolic + diastolic ---
    bp_split = df["Blood Pressure"].str.split("/", expand=True)
    df["Systolic BP"]  = pd.to_numeric(bp_split[0], errors="coerce")
    df["Diastolic BP"] = pd.to_numeric(bp_split[1], errors="coerce")
    df.drop(columns=["Blood Pressure"], inplace=True)

    # --- Ordinal encode BMI Category (Normal < Overweight < Obese) ---
    bmi_order = {"Normal": 0, "Normal Weight": 0, "Overweight": 1, "Obese": 2}
    df["BMI Category"] = df["BMI Category"].map(bmi_order)

    print("Preprocessed shape:", df.shape)
    print("Class distribution after fill:\n", df["Sleep Disorder"].value_counts())
    print()

    return df


# =============================================================================
# 4. FEATURE / TARGET SPLIT & TRAIN–TEST SPLIT
# =============================================================================

def split_data(df: pd.DataFrame, target: str = "Sleep Disorder", test_size: float = 0.2,
               random_state: int = 42):
    """
    Separate feature matrix X and label vector y.
    Uses a stratified split to maintain class proportions.
    """
    X = df.drop(columns=[target])
    y = df[target]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols   = X.select_dtypes(exclude="object").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training samples : {X_train.shape[0]}")
    print(f"Test samples     : {X_test.shape[0]}")
    print(f"Numerical cols   : {numerical_cols}")
    print(f"Categorical cols : {categorical_cols}")
    print()

    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols


# =============================================================================
# 5. SKLEARN PIPELINE (ColumnTransformer)
# =============================================================================

def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Imputes and standardises numerical features
      - Imputes and one-hot-encodes categorical features (Gender, Occupation)
    """
    from sklearn.preprocessing import OneHotEncoder

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


# =============================================================================
# 6. MODEL TRAINING & EVALUATION
# =============================================================================

def evaluate_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test,
                   cv_folds: int = 5):
    """
    Fit the pipeline on training data, evaluate on the held-out test set,
    and report accuracy, precision, recall, F1-score, and confusion matrix.
    Also reports stratified k-fold cross-validation mean ± std.
    """
    # Train
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf,
                                scoring="f1_weighted", n_jobs=-1)

    print(f"\n{'=' * 60}")
    print(f"MODEL: {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy        : {accuracy:.4f}")
    print(f"  Precision (W)   : {precision:.4f}")
    print(f"  Recall (W)      : {recall:.4f}")
    print(f"  F1-Score (W)    : {f1:.4f}")
    print(f"  CV F1 (mean±std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return {
        "model": name, "pipeline": pipeline,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
    }


# =============================================================================
# 7. HYPERPARAMETER TUNING (Random Forest)
# =============================================================================

def tune_random_forest(preprocessor, X_train, y_train) -> Pipeline:
    """
    Perform GridSearchCV over key Random Forest hyperparameters.
    Uses 5-fold stratified cross-validation with weighted F1 as the
    scoring metric to handle class imbalance.
    """
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])

    param_grid = {
        "classifier__n_estimators"  : [100, 200, 300],
        "classifier__max_depth"     : [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__max_features"  : ["sqrt", "log2"]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nRunning GridSearchCV for Random Forest (this may take a moment)...")
    grid_search = GridSearchCV(
        rf_pipeline, param_grid,
        cv=skf, scoring="f1_weighted",
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best parameters : {grid_search.best_params_}")
    print(f"  Best CV F1      : {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# =============================================================================
# 8. MODEL COMPARISON TABLE
# =============================================================================

def print_comparison(results: list):
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    header = f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'CV F1':>10}"
    print(header)
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<22} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f} "
              f"{r['cv_mean']:>8.4f}±{r['cv_std']:.3f}")
    print("=" * 70)
    best = max(results, key=lambda x: x["cv_mean"])
    print(f"\n  ✔  Recommended final model: {best['model']} (CV F1 = {best['cv_mean']:.4f})")


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    # ------------------------------------------------------------------ #
    # Step 1 – Load data
    # ------------------------------------------------------------------ #
    filepath = "Sleep_health_and_lifestyle_dataset.csv"
    df = load_and_inspect(filepath)

    # ------------------------------------------------------------------ #
    # Step 2 – Preprocess
    # ------------------------------------------------------------------ #
    df = preprocess(df)

    # ------------------------------------------------------------------ #
    # Step 3 – Split
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test, num_cols, cat_cols = split_data(df)

    # ------------------------------------------------------------------ #
    # Step 4 – Build preprocessor
    # ------------------------------------------------------------------ #
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # ------------------------------------------------------------------ #
    # Step 5 – Define models
    # ------------------------------------------------------------------ #
    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor(num_cols, cat_cols)),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced", random_state=42
            ))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_preprocessor(num_cols, cat_cols)),
            ("classifier", DecisionTreeClassifier(
                max_depth=10, class_weight="balanced", random_state=42
            ))
        ]),
        "Random Forest (default)": Pipeline([
            ("preprocessor", build_preprocessor(num_cols, cat_cols)),
            ("classifier", RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42
            ))
        ])
    }

    # ------------------------------------------------------------------ #
    # Step 6 – Evaluate baseline models
    # ------------------------------------------------------------------ #
    results = []
    for name, pipeline in models.items():
        result = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(result)

    # ------------------------------------------------------------------ #
    # Step 7 – Tune Random Forest
    # ------------------------------------------------------------------ #
    tuned_rf = tune_random_forest(build_preprocessor(num_cols, cat_cols), X_train, y_train)
    tuned_result = evaluate_model(
        "Random Forest (tuned)", tuned_rf,
        X_train, X_test, y_train, y_test
    )
    results.append(tuned_result)

    # ------------------------------------------------------------------ #
    # Step 8 – Summary
    # ------------------------------------------------------------------ #
    print_comparison(results)

    print("\nTraining complete. Use app.py for the Streamlit deployment.")


if __name__ == "__main__":
    main()
