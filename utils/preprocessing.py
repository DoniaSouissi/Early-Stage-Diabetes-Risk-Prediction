import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


def clean_duplicates(df: pd.DataFrame):
    """
    Drops duplicate rows from the DataFrame and returns the clean version.
    Prints the number of rows removed.
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates()
    final_count = len(df_clean)
    
    print(f" Dropped {initial_count - final_count} duplicate rows.")
    print(f"New Data Shape: {df_clean.shape}")
    
    return df_clean

def detect_outliers_iqr(df, column):
    """
    Detects outliers in a numerical column using the Interquartile Range (IQR) method.
    
    Returns:
    - lower_bound (float): The lower cutoff limit.
    - upper_bound (float): The upper cutoff limit.
    - num_outliers (int): The count of outliers found.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    num_outliers = len(outliers)
    
    return lower_bound, upper_bound, num_outliers



def split_data(df, target_column='class', test_size=0.2, val_size=0.2, random_state=42):
    """
    Splits the data into Train (60%), Validation (20%), and Test (20%).
    
    Returns:
    - X_train, X_val, X_test (DataFrames)
    - y_train, y_val, y_test (Series)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 1. Split Train (60%) vs Rest (40%)
    # Logic: If we want 60% train, we set test_size to 0.4 (which is Val+Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    
    # 2. Split Rest (40%) into Val (20%) and Test (20%)
    # Logic: Since X_temp is 40% of original, splitting 0.5 gives 20% total each.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_preprocessor():
    """
    Creates and returns a Scikit-Learn ColumnTransformer.
    - Applies StandardScaler to 'Age'.
    - Applies OneHotEncoder to all Categorical symptoms.
    """
    # 1. Define Features
    numerical_features = ['Age']
    
    # All categorical columns from the UCI Dataset
    categorical_features = [
        'Polyuria', 'Polydipsia', 'sudden weight loss',
        'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
        'Itching', 'Irritability', 'delayed healing', 'partial paresis',
        'muscle stiffness', 'Alopecia', 'Obesity'
    ]
    
    # 2. Create Transformers
    numerical_transformer = StandardScaler()
    
    # drop='first' avoids multicollinearity (dummy variable trap)
    # handle_unknown='ignore' prevents crashing if a new category appears in production
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    
    # 3. Bundle into Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def encode_target(y_train, y_val, y_test):
    """
    Encodes the target variable (Positive/Negative) into integers (1/0).
    Returns the transformed targets and the fitted encoder.
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    
    return y_train_enc, y_val_enc, y_test_enc, le

def save_artifacts(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, label_encoder):
    """
    Saves the processed datasets and model artifacts (joblib files).
    """
    # Ensure directories exist
    os.makedirs('../data/processed', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Save CSVs
    X_train.to_csv('../data/processed/X_train.csv', index=False)
    X_val.to_csv('../data/processed/X_val.csv', index=False)
    X_test.to_csv('../data/processed/X_test.csv', index=False)
    
    # Save Targets
    pd.DataFrame(y_train, columns=['class']).to_csv('../data/processed/y_train.csv', index=False)
    pd.DataFrame(y_val, columns=['class']).to_csv('../data/processed/y_val.csv', index=False)
    pd.DataFrame(y_test, columns=['class']).to_csv('../data/processed/y_test.csv', index=False)
    
    # Save Artifacts
    joblib.dump(preprocessor, '../models/preprocessor.joblib')
    joblib.dump(label_encoder, '../models/target_encoder.joblib')
    
    print(" All files and models saved successfully!")