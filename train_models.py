"""
Machine Learning Training Pipeline
Customer Behaviour Analysis & Churn Prediction
Compares Logistic Regression, Decision Tree, Random Forest, and KNN.
Trains K-Means for Customer Segmentation.
Saves models and evaluation results.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    load_raw_data,
    clean_data,
    add_engineered_features,
    build_preprocessor,
    prepare_full_pipeline
)
from src.evaluation import evaluate_model, cross_validate_model, compare_models


def run_training_pipeline():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_dir, 'dataset', 'customer_churn.csv')
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("Step 1: Loading raw data...")
    raw_df = load_raw_data(data_path)
    print(f"Loaded {len(raw_df)} records.")

    print("Step 2: Cleaning and engineering features...")
    X, y, preprocessor, feat_df = prepare_full_pipeline(raw_df)
    print(f"Features shape: {X.shape}, Target distribution: \n{y.value_counts(normalize=True)}")

    print("Step 3: Splitting train and test sets (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Define Candidate Classifiers
    candidate_classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs',
            C=1.0
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight='balanced',
            random_state=42,
            min_samples_split=20
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight='balanced',
            random_state=42,
            min_samples_leaf=5,
            n_jobs=-1
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            n_jobs=-1
        )
    }

    print("\nStep 4: Training & Evaluating Algorithms...")
    trained_pipelines = {}
    evaluation_summary = {}
    cv_summary = {}

    for name, clf in candidate_classifiers.items():
        print(f"  --> Training {name}...")
        pipe = Pipeline(steps=[
            ('preprocessor', build_preprocessor()),
            ('classifier', clf)
        ])
        
        # Cross validation
        cv_res = cross_validate_model(pipe, X_train, y_train, cv=5)
        cv_summary[name] = cv_res

        # Fit on full training set
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Test set evaluation
        test_eval = evaluate_model(pipe, X_test, y_test)
        evaluation_summary[name] = test_eval

        print(f"      [Test] Accuracy: {test_eval['accuracy']:.4f} | Recall: {test_eval['recall']:.4f} | ROC-AUC: {test_eval['roc_auc']:.4f}")

    # Generate comparison table
    df_comparison = compare_models(trained_pipelines, X_test, y_test)
    print("\n--- Model Comparison on Test Set ---")
    print(df_comparison.to_string(index=False))

    # Save comparison dataframe to CSV
    df_comparison.to_csv(os.path.join(models_dir, 'model_comparison.csv'), index=False)

    # Feature Importance from Random Forest
    rf_pipe = trained_pipelines["Random Forest"]
    rf_clf = rf_pipe.named_steps['classifier']
    preproc = rf_pipe.named_steps['preprocessor']

    feature_names = preproc.get_feature_names_out()
    importances = rf_clf.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    feat_imp_df.to_csv(os.path.join(models_dir, 'feature_importance.csv'), index=False)

    print("\nTop 10 Most Important Features (Random Forest):")
    print(feat_imp_df.head(10).to_string(index=False))

    # Select Best Model for Deployment: Random Forest
    best_model_name = "Random Forest"
    best_pipeline = trained_pipelines[best_model_name]
    best_model_path = os.path.join(models_dir, 'churn_model.pkl')
    joblib.dump(best_pipeline, best_model_path)
    print(f"\nBest model ({best_model_name}) saved to {best_model_path}")

    # Step 5: Customer Segmentation with K-Means
    print("\nStep 5: Fitting K-Means Customer Segmentation...")
    # Segmentation using tenure and MonthlyCharges
    cluster_features = ['tenure', 'MonthlyCharges']
    clustering_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=4, random_state=42, n_init=10))
    ])
    clustering_pipeline.fit(feat_df[cluster_features])
    
    clustering_model_path = os.path.join(models_dir, 'clustering_model.pkl')
    joblib.dump(clustering_pipeline, clustering_model_path)
    print(f"K-Means model saved to {clustering_model_path}")

    # Compute segment summaries
    feat_df['Cluster'] = clustering_pipeline.predict(feat_df[cluster_features])
    segment_names = {
        0: "Low Spending, New Customers (Budget Starters)",
        1: "High Spending, Long-Term (Loyal Champions)",
        2: "High Spending, Short-Term (High Risk Vulnerables)",
        3: "Low Spending, Long-Term (Steady Traditionalists)"
    }
    feat_df['Segment_Name'] = feat_df['Cluster'].map(segment_names)
    
    seg_summary = feat_df.groupby(['Cluster', 'Segment_Name']).agg(
        Customer_Count=('tenure', 'count'),
        Avg_Tenure=('tenure', 'mean'),
        Avg_Monthly_Charges=('MonthlyCharges', 'mean'),
        Churn_Rate=('Churn', 'mean')
    ).reset_index()
    seg_summary.to_csv(os.path.join(models_dir, 'segment_summary.csv'), index=False)
    print("\nCustomer Segments Summary:")
    print(seg_summary.to_string(index=False))

    # Step 6: Serialize evaluation metadata for Dashboard
    # Convert non-serializable elements
    eval_export = {}
    for name, metrics in evaluation_summary.items():
        eval_export[name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'confusion_matrix': metrics['confusion_matrix'],
            'fpr': metrics['fpr'],
            'tpr': metrics['tpr'],
            'cv_mean_roc_auc': cv_summary[name]['cv_roc_auc_mean'],
            'cv_mean_recall': cv_summary[name]['cv_recall_mean']
        }

    with open(os.path.join(models_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_export, f, indent=4)

    print("\nAll training and segmentation steps completed successfully!")


if __name__ == "__main__":
    run_training_pipeline()
