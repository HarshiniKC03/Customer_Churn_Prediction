"""
Model Evaluation and Metrics Module
Customer Behaviour Analysis & Churn Prediction
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Computes comprehensive evaluation metrics for a trained model on test data.
    """
    y_pred = model.predict(X_test)
    
    # Check if model provides predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    else:
        y_prob = y_pred
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr = None, None
        prec_curve, rec_curve = None, None

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(auc),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'y_prob': y_prob.tolist() if isinstance(y_prob, np.ndarray) else y_prob,
        'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
        'fpr': fpr.tolist() if fpr is not None else None,
        'tpr': tpr.tolist() if tpr is not None else None,
    }
    return metrics


def cross_validate_model(model, X, y, cv=5) -> dict:
    """
    Performs Stratified K-Fold cross validation and returns mean and std of metrics.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    
    summary = {
        'cv_accuracy_mean': float(np.mean(cv_results['test_accuracy'])),
        'cv_accuracy_std': float(np.std(cv_results['test_accuracy'])),
        'cv_precision_mean': float(np.mean(cv_results['test_precision'])),
        'cv_precision_std': float(np.std(cv_results['test_precision'])),
        'cv_recall_mean': float(np.mean(cv_results['test_recall'])),
        'cv_recall_std': float(np.std(cv_results['test_recall'])),
        'cv_f1_mean': float(np.mean(cv_results['test_f1'])),
        'cv_f1_std': float(np.std(cv_results['test_f1'])),
        'cv_roc_auc_mean': float(np.mean(cv_results['test_roc_auc'])),
        'cv_roc_auc_std': float(np.std(cv_results['test_roc_auc'])),
    }
    return summary


def compare_models(models_dict: dict, X_test, y_test) -> pd.DataFrame:
    """
    Generates comparison DataFrame across all candidate models.
    """
    records = []
    for name, model in models_dict.items():
        res = evaluate_model(model, X_test, y_test)
        records.append({
            'Model': name,
            'Accuracy': round(res['accuracy'], 4),
            'Precision (Churn)': round(res['precision'], 4),
            'Recall (Churn)': round(res['recall'], 4),
            'F1-Score': round(res['f1_score'], 4),
            'ROC-AUC': round(res['roc_auc'], 4),
            'True Positives': res['confusion_matrix']['tp'],
            'False Negatives': res['confusion_matrix']['fn'],
            'False Positives': res['confusion_matrix']['fp'],
            'True Negatives': res['confusion_matrix']['tn']
        })
    df_compare = pd.DataFrame(records).sort_values(by='Recall (Churn)', ascending=False)
    return df_compare

