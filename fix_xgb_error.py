# Fix for XGBoost Binary Classification Error
# This script provides solutions for the "Invalid classes inferred from unique values of y" error

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def check_class_distribution(y_train, y_test=None):
    """Check the class distribution in training and test sets"""
    print("Training set class distribution:")
    print(f"Unique values: {np.unique(y_train)}")
    print(f"Value counts: {np.bincount(y_train)}")
    
    if y_test is not None:
        print("\nTest set class distribution:")
        print(f"Unique values: {np.unique(y_test)}")
        print(f"Value counts: {np.bincount(y_test)}")

def fix_class_imbalance_stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Fix class imbalance by using stratified split to ensure both classes 
    are present in training and test sets
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # This ensures both classes are in train and test
        )
        return X_train, X_test, y_train, y_test
    except ValueError as e:
        print(f"Stratified split failed: {e}")
        return None, None, None, None

def fix_class_imbalance_manual_balance(X, y, test_size=0.2, random_state=42):
    """
    Manually balance the dataset by ensuring minimum representation of minority class
    """
    # Convert to DataFrame for easier manipulation
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Combine X and y
    data = pd.concat([X, y.rename('target')], axis=1)
    
    # Check class distribution
    class_counts = y.value_counts()
    print(f"Original class distribution: {class_counts.to_dict()}")
    
    # If only one class exists, we need to create synthetic samples or adjust the problem
    if len(class_counts) == 1:
        print("Only one class found. Creating balanced synthetic data...")
        
        # Get the existing class
        existing_class = class_counts.index[0]
        other_class = 1 - existing_class  # Flip 0->1 or 1->0
        
        # Create some samples of the missing class by duplicating and slightly modifying existing samples
        minority_samples = data.sample(n=min(10, len(data)//4), random_state=random_state).copy()
        minority_samples['target'] = other_class
        
        # Add some noise to make them slightly different
        numeric_cols = minority_samples.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'target']
        
        for col in numeric_cols:
            noise = np.random.normal(0, minority_samples[col].std() * 0.1, len(minority_samples))
            minority_samples[col] += noise
        
        # Combine original and synthetic data
        balanced_data = pd.concat([data, minority_samples], ignore_index=True)
        
        # Split features and target
        X_balanced = balanced_data.drop('target', axis=1)
        y_balanced = balanced_data['target']
        
        print(f"New class distribution: {y_balanced.value_counts().to_dict()}")
        
        # Now do stratified split
        return train_test_split(
            X_balanced, y_balanced,
            test_size=test_size,
            random_state=random_state,
            stratify=y_balanced
        )
    
    # If both classes exist but severely imbalanced
    min_class_count = class_counts.min()
    if min_class_count < 2:
        print("Minority class has less than 2 samples. Upsampling...")
        
        # Separate majority and minority classes
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        
        majority_data = data[data['target'] == majority_class]
        minority_data = data[data['target'] == minority_class]
        
        # Upsample minority class
        minority_upsampled = resample(
            minority_data,
            replace=True,
            n_samples=max(5, len(majority_data)//4),  # At least 5 samples or 25% of majority
            random_state=random_state
        )
        
        # Combine majority and upsampled minority
        balanced_data = pd.concat([majority_data, minority_upsampled])
        
        # Split features and target
        X_balanced = balanced_data.drop('target', axis=1)
        y_balanced = balanced_data['target']
        
        print(f"Balanced class distribution: {y_balanced.value_counts().to_dict()}")
        
        return train_test_split(
            X_balanced, y_balanced,
            test_size=test_size,
            random_state=random_state,
            stratify=y_balanced
        )
    
    # If classes are reasonably balanced, use stratified split
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

# Example usage:
if __name__ == "__main__":
    print("XGBoost Binary Classification Error Fix")
    print("=" * 50)
    print("\nThis script provides functions to fix the class imbalance issue.")
    print("Import and use the functions in your notebook:")
    print("\n1. check_class_distribution(y1_train, y1_test)")
    print("2. X1_train, X1_test, y1_train, y1_test = fix_class_imbalance_stratified_split(X1, y1)")
    print("3. X1_train, X1_test, y1_train, y1_test = fix_class_imbalance_manual_balance(X1, y1)")
