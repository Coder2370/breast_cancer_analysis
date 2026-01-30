"""
Extended analysis of Breast Cancer Wisconsin (Diagnostic) dataset.

This script loads the dataset from scikit-learn, saves it to an SQLite database,
computes summary statistics, plots a correlation matrix and trains several classifiers:
Logistic Regression, Random Forest, Decision Tree, K-Nearest Neighbours and SVC.
It also performs cross-validation to compare model performance. The dataset has
569 samples with 30 real-valued features and a binary target indicating
malignant (0) or benign (1) tumours
"""

import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def load_dataset():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={'target': 'diagnosis'}, inplace=True)
    df['diagnosis_label'] = df['diagnosis'].map({0: 'malignant', 1: 'benign'})
    return df


def save_to_sqlite(df, db_path=None):
    if db_path is None:
        script_dir = Path(__file__).resolve().parent
        db_path = script_dir / 'breast_cancer.db'
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql('breast_cancer', conn, if_exists='replace', index=False)
    finally:
        conn.close()


def plot_correlation_matrix(df, output_path=None):
    if output_path is None:
        script_dir = Path(__file__).resolve().parent
        output_path = script_dir / 'correlation_heatmap.png'
    feature_df = df.drop(columns=['diagnosis', 'diagnosis_label'])
    corr = feature_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True,
                cbar_kws={'shrink': 0.8}, xticklabels=True, yticklabels=True)
    plt.title('Correlation Matrix of Breast Cancer Features')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_and_evaluate_logistic(df, seed=42):
    X = df.drop(columns=['diagnosis', 'diagnosis_label'])
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_pred_proba)
    print('Logistic Regression accuracy:', acc)
    print('Logistic Regression AUC:', auc)
    return model, acc, auc


def build_and_evaluate_random_forest(df, seed=42):
    X = df.drop(columns=['diagnosis', 'diagnosis_label'])
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    acc = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_pred_proba)
    print('Random Forest accuracy:', acc)
    print('Random Forest AUC:', auc)
    return model, acc, auc


def build_and_evaluate_decision_tree(df, seed=42):
    from sklearn.tree import DecisionTreeClassifier
    X = df.drop(columns=['diagnosis', 'diagnosis_label'])
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    model = DecisionTreeClassifier(random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    acc = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_pred_proba)
    print('Decision Tree accuracy:', acc)
    print('Decision Tree AUC:', auc)
    return model, acc, auc


def build_and_evaluate_knn(df, n_neighbors=5, seed=42):
    from sklearn.neighbors import KNeighborsClassifier
    X = df.drop(columns=['diagnosis', 'diagnosis_label'])
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'KNN (k={n_neighbors}) accuracy:', acc)
    print(f'KNN (k={n_neighbors}) AUC:', auc)
    return model, acc, auc


def build_and_evaluate_svc(df, C=1.0, kernel='rbf', seed=42):
    from sklearn.svm import SVC
    X = df.drop(columns=['diagnosis', 'diagnosis_label'])
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(C=C, kernel=kernel, probability=True, random_state=seed)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'SVC ({kernel} kernel) accuracy:', acc)
    print(f'SVC ({kernel} kernel) AUC:', auc)
    return model, acc, auc


def cross_validate_models(df, seed=42, cv=5):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    X = df.drop(columns=['diagnosis', 'diagnosis_label'])
    y = df['diagnosis']
    models = {
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000, random_state=seed))]),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=seed),
        'Decision Tree': DecisionTreeClassifier(random_state=seed),
        'KNN': Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=5))]),
        'SVC': Pipeline([('scaler', StandardScaler()), ('model', SVC(C=1.0, kernel='rbf', probability=True, random_state=seed))])
    }
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f'{name} cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}')


def main():
    df = load_dataset()
    save_to_sqlite(df)
    plot_correlation_matrix(df)
    build_and_evaluate_logistic(df)
    build_and_evaluate_random_forest(df)
    build_and_evaluate_decision_tree(df)
    build_and_evaluate_knn(df)
    build_and_evaluate_svc(df)
    cross_validate_models(df)


if __name__ == '__main__':
    main()
