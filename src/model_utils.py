# src/model_utils.py

# Import necessary libraries and packages
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(file_path= 'data/processed_cleaned_data.csv'):
    """
    Load the processed dataset from a CSV file

    Args:
        file_path (str): Relative path to the processed CSV.
        
    Returns:
        pd.DataFrame: Loaded Data.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Data fiel not found at {file_path}')
    return pd.read_csv(file_path)

def split_data(df, target_column= 'Defaulter', test_size= 0.2, random_state= 42, stratify= True):
    """
    Split the DataFrame into training and testing sets.
    
    Args:
        df (pd.DataFrame): The DataFrame to split.
        target_column (str): Name of the target variable column.
        test_size (float): Proportion of the data to include in the test split.
        random_state (int): Seed used by the random number generator.
        stratify (bool): Whether to stratify by the target variable.
        
    Returns:
        X_train, X_test, y_train, y_test: The split features and target sets.
    """

    X= df.drop(columns= [target_column])
    y= df[target_column]
    stratify_var= y if stratify else None
    X_train, y_train, X_test, y_test= train_test_split(X, y, test_size= test_size, random_state= random_state, stratify= stratify_var)
    print(f'Training set shape: {X_train.shape}, Test set shape: {X_test.shape}')

    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred):
     """
    Print evaluation metrics for the model's predictions.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    """
     
     
     print('=== Classification Report ===')
     print(classification_report(y_true, y_pred))
     print('=== Confusion Matrix ===')
     print(confusion_matrix(y_true, y_pred))
     print('Accuracy:', accuracy_score(y_true, y_pred))
    