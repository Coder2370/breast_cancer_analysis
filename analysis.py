"""
Breast Cancer Wisconsin (Diagnostic) data analysis and modeling
================================================================

This script performs exploratory data analysis (EDA) and predictive modeling
on the Breast Cancer Wisconsin (Diagnostic) dataset, which is included in
scikit‑learn.  The dataset contains measurements computed from fine‑needle
aspirate (FNA) images of breast masses.  Each of the 569 samples has 30
numeric features and a binary target indicating whether the tumour is
malignant (0) or benign (1).  According to the scikit‑learn documentation,
the dataset has two classes (malignant and benign), with 212 malignant
samples and 357 benign samples, giving 569 samples in total and 30
positive‑valued features.

The goals of this script are:

1. Load the dataset into a pandas DataFrame and store it in an SQLite database
   for easy querying via SQL.
2. Use SQL queries to compute summary statistics such as the average of
   selected features grouped by diagnosis.
3. Perform EDA in Python, including computing a correlation matrix and
   visualising it as a heatmap.
4. Build a logistic regression classifier to predict tumour malignancy from
   the available features, evaluate it using accuracy and a classification
   report, and display the resulting ROC curve.

Running this script will create a SQLite database file named
``breast_cancer.db`` in the project directory and save a plot of the
correlation matrix and ROC curve as PNG files.

Usage::

    python analysis.py

Dependencies: pandas, numpy, matplotlib, seaborn, scikit‑learn, sqlite3
"""

import sqlite3
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def load_dataset() -> pd.DataFrame:
    """Load the Breast Cancer Wisconsin dataset into a DataFrame.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing 569 rows and 31 columns: the 30
        feature columns plus a 'target' column indicating diagnosis (0 =
        malignant, 1 = benign).
    """
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # Rename the target column to something more meaningful
    df.rename(columns={'target': 'diagnosis'}, inplace=True)
    # Map diagnosis from 0/1 to human‑readable labels
    df['diagnosis_label'] = df['diagnosis'].map({0: 'malignant', 1: 'benign'})
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: Path = None) -> None:
    """Store the DataFrame in an SQLite database.

    Parameters
    ----------
    df : DataFrame
        The breast cancer data.
    db_path : Path
        Location where the SQLite database should be saved.
    """
    # Connect to SQLite (will create file if it doesn't exist)
    # Default to a database within the script's directory if none provided
    if db_path is None:
        script_dir = Path(__file__).resolve().parent
        db_path = script_dir / 'breast_cancer.db'
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql('breast_cancer', conn, if_exists='replace', index=False)
    finally:
        conn.close()


def query_summary_stats(db_path: Path = None) -> pd.DataFrame:
    """Run a SQL query to compute average feature values by diagnosis.

    This function computes the mean of select features for malignant and benign
    tumours using SQL.  It demonstrates how SQL can be used for summarising
    data stored in a relational database.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database.

    Returns
    -------
    DataFrame
        A pandas DataFrame with one row per diagnosis and columns for the
        average of selected features.
    """
    # Default path: database in the same directory as this script
    if db_path is None:
        script_dir = Path(__file__).resolve().parent
        db_path = script_dir / 'breast_cancer.db'
    conn = sqlite3.connect(db_path)
    try:
        query = """
        SELECT
            diagnosis_label,
            AVG(`mean radius`) AS avg_mean_radius,
            AVG(`mean texture`) AS avg_mean_texture,
            AVG(`mean area`) AS avg_mean_area,
            AVG(`mean smoothness`) AS avg_mean_smoothness
        FROM breast_cancer
        GROUP BY diagnosis_label
        ORDER BY diagnosis_label;
        """
        summary_df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return summary_df


def plot_correlation_matrix(df: pd.DataFrame, output_path: Path = None) -> None:
    """Plot and save the correlation matrix heatmap for the features.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing feature columns and the diagnosis.
    output_path : Path
        File path to save the plot image (PNG).
    """
    # Determine default output path relative to the script
    if output_path is None:
        script_dir = Path(__file__).resolve().parent
        output_path = script_dir / 'correlation_heatmap.png'

    # Exclude diagnosis columns from correlation calculation
    feature_df = df.drop(columns=['diagnosis', 'diagnosis_label'])
    corr = feature_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True,
                cbar_kws={'shrink': 0.8}, xticklabels=True, yticklabels=True)
    plt.title('Correlation Matrix of Breast Cancer Features')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_and_evaluate_model(df: pd.DataFrame, seed: int = 42) -> Tuple[LogisticRegression, float, float]:
    """Build a logistic regression classifier and evaluate it.

    Parameters
    ----------
    df : DataFrame
        The breast cancer dataset with features and diagnosis.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    model : LogisticRegression
        The trained logistic regression model.
    accuracy : float
        The classification accuracy on the test set.
    auc : float
        The area under the ROC curve on the test set.
    """
    # Separate features and target
    X = df.drop(columns=['diagnosis', 'diagnosis_label'])
    y = df['diagnosis']
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    # Standardise features to zero mean and unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train_scaled, y_train)
    # Predict probabilities for ROC curve
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    # Print classification report to console
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['malignant', 'benign']))
    # Plot ROC curve
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_pred_proba)
    plt.title('ROC Curve for Logistic Regression')
    plt.tight_layout()
    # Save ROC figure relative to script directory
    script_dir = Path(__file__).resolve().parent
    roc_path = script_dir / 'roc_curve.png'
    plt.savefig(roc_path)
    plt.close()
    return model, accuracy, auc


def main() -> None:
    """Main entry point of the script."""
    # Load dataset and save to SQL
    df = load_dataset()
    save_to_sqlite(df)
    # Run SQL summary statistics
    summary = query_summary_stats()
    print("\nAverage feature values by diagnosis (SQL):")
    print(summary)
    # Plot correlation matrix
    plot_correlation_matrix(df)
    # Build model and evaluate
    model, accuracy, auc = build_and_evaluate_model(df)
    print(f"\nModel accuracy on test set: {accuracy:.4f}")
    print(f"Area under ROC curve: {auc:.4f}\n")


if __name__ == '__main__':
    main()
