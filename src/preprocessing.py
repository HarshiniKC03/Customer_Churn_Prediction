"""
Data Preprocessing and Feature Engineering Module
Customer Behaviour Analysis & Churn Prediction
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


NUMERICAL_COLS = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'ServiceCount',
    'AvgMonthlyCharges'
]

CATEGORICAL_COLS = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'TenureGroup',
    'SecurityBundle',
    'StreamingBundle'
]


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw Telco dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Converts TotalCharges to numeric, filling whitespace or NaN with 0.0 (tenure=0 cases)
    - Drops customerID if present
    - Maps target Churn column to binary (1 for Yes, 0 for No) if present
    """
    data = df.copy()

    # Drop customerID if exists
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])

    # Handle TotalCharges blanks
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        # TotalCharges is NaN when tenure is 0 (new customers); fill with 0.0
        data['TotalCharges'] = data['TotalCharges'].fillna(0.0)

    # Convert SeniorCitizen to string or object for consistent categorical handling
    if 'SeniorCitizen' in data.columns:
        data['SeniorCitizen'] = data['SeniorCitizen'].astype(str)

    # Encode target if present
    if 'Churn' in data.columns:
        if data['Churn'].dtype == object:
            data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

    return data


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates domain-specific features for customer behaviour analysis:
    1. TenureGroup: New (<=12m), Regular (13-36m), Loyal (>36m)
    2. ServiceCount: Total number of active subscribed services
    3. AvgMonthlyCharges: Historical monthly expenditure rate
    4. SecurityBundle: Both OnlineSecurity and TechSupport active
    5. StreamingBundle: Both StreamingTV and StreamingMovies active
    """
    data = df.copy()

    # 1. Tenure Group
    def categorize_tenure(months):
        if months <= 12:
            return 'New Customer'
        elif months <= 36:
            return 'Regular Customer'
        else:
            return 'Loyal Customer'

    if 'tenure' in data.columns:
        data['TenureGroup'] = data['tenure'].apply(categorize_tenure)

    # 2. Service Count
    service_cols = [
        'PhoneService',
        'MultipleLines',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]
    
    count = pd.Series(0, index=data.index)
    for col in service_cols:
        if col in data.columns:
            count += (data[col] == 'Yes').astype(int)
    
    # Also add InternetService if not 'No'
    if 'InternetService' in data.columns:
        count += (data['InternetService'].isin(['DSL', 'Fiber optic'])).astype(int)

    data['ServiceCount'] = count

    # 3. AvgMonthlyCharges
    if 'TotalCharges' in data.columns and 'tenure' in data.columns:
        # Avoid division by zero for tenure=0
        safe_tenure = np.where(data['tenure'] == 0, 1, data['tenure'])
        data['AvgMonthlyCharges'] = np.round(data['TotalCharges'] / safe_tenure, 2)
    elif 'MonthlyCharges' in data.columns:
        data['AvgMonthlyCharges'] = data['MonthlyCharges']

    # 4. Security Bundle (OnlineSecurity + TechSupport)
    if 'OnlineSecurity' in data.columns and 'TechSupport' in data.columns:
        data['SecurityBundle'] = np.where(
            (data['OnlineSecurity'] == 'Yes') & (data['TechSupport'] == 'Yes'),
            'Yes', 'No'
        )
    else:
        data['SecurityBundle'] = 'No'

    # 5. Streaming Bundle (StreamingTV + StreamingMovies)
    if 'StreamingTV' in data.columns and 'StreamingMovies' in data.columns:
        data['StreamingBundle'] = np.where(
            (data['StreamingTV'] == 'Yes') & (data['StreamingMovies'] == 'Yes'),
            'Yes', 'No'
        )
    else:
        data['StreamingBundle'] = 'No'

    return data


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-learn ColumnTransformer for numerical scaling
    and categorical one-hot encoding.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ],
        remainder='drop'
    )
    return preprocessor


def prepare_full_pipeline(df: pd.DataFrame):
    """
    Preprocesses raw dataframe, adds engineered features, and splits into X, y.
    Returns (X, y, preprocessor).
    """
    clean_df = clean_data(df)
    feat_df = add_engineered_features(clean_df)

    if 'Churn' in feat_df.columns:
        X = feat_df.drop(columns=['Churn'])
        y = feat_df['Churn']
    else:
        X = feat_df
        y = None

    preprocessor = build_preprocessor()
    return X, y, preprocessor, feat_df

