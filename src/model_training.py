# src/model_training.py

# Import necessary libraries and packages
import joblib
import numpy as np
import pyspark
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from src.model_utils import load_data, split_data, evaluate_model
from hyperopt import fmin, tpe, hp, STATUS_OK, SparkTrials

def tune_xgboost(X_tr, y_tr, X_val, y_val, max_evals=50):
    """
    Tune XGBoost hyperparameters using a fixed train-validation split and SparkTrials for parallel evaluations.
    Returns the best hyperparameters as a dictionary.
    """
    def objective(params):
        clf = xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        score = f1_score(y_val, y_pred)
        return {'loss': 1 - score, 'status': STATUS_OK}
    
    space = {
        'max_depth': hp.choice('max_depth', list(range(3, 15))),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
        'n_estimators': hp.choice('n_estimators', list(range(100, 501, 50))),
        'gamma': hp.uniform('gamma', 0, 5),
        'min_child_weight': hp.uniform('min_child_weight', 1, 10),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
    }
    
    # Use SparkTrials for parallelism
    spark_trials = SparkTrials(parallelism=16)
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=spark_trials,
        rstate=np.random.default_rng(42)
    )
    
    best_params = {
        'max_depth': list(range(3, 15))[best['max_depth']],
        'learning_rate': best['learning_rate'],
        'n_estimators': list(range(100, 501, 50))[best['n_estimators']],
        'gamma': best['gamma'],
        'min_child_weight': best['min_child_weight'],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree']
    }
    return best_params

def tune_random_forest(X_tr, y_tr, X_val, y_val, max_evals=50):
    """
    Tune Random Forest hyperparameters using a fixed train-validation split and SparkTrials for parallel evaluations.
    Returns the best hyperparameters as a dictionary.
    """
    def objective(params):
        # hp.quniform returns floats; convert these to int
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        score = f1_score(y_val, y_pred)
        return {'loss': 1 - score, 'status': STATUS_OK}
    
    space_rf = {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500]),
        'max_depth': hp.choice('max_depth', [None] + list(range(5, 21))),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
    }
    
    spark_trials = SparkTrials(parallelism=16)
    best = fmin(
        fn=objective,
        space=space_rf,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=spark_trials,
        rstate=np.random.default_rng(42)
    )
    
    best_params = {
        'n_estimators': [100, 200, 300, 400, 500][best['n_estimators']],
        'max_depth': ([None] + list(range(5, 21)))[best['max_depth']],
        'min_samples_split': int(best['min_samples_split']),
        'min_samples_leaf': int(best['min_samples_leaf']),
        'max_features': ['sqrt', 'log2', None][best['max_features']]
    }
    return best_params

def main():
    print("Start model training...")

    # Load the processed data
    df = load_data()
    
    # Drop non-encoded columns that can cause issues 
    df = df.drop(columns=['Industry', 'Work Experience', 'Role'], errors='ignore')
    
    # Split the data into training and test sets using 'Defaulter' as the target
    X_train, y_train, X_test, y_test = split_data(df, target_column='Defaulter')
    
    # Pre-split the training set into a tuning split and a fixed validation set
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # -------------------------------
    # Baseline Random Forest Model
    # -------------------------------
    print("Training baseline Random Forest model...")
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_baseline.fit(X_train, y_train)
    y_pred_rf_base = rf_baseline.predict(X_test)
    print("\nBaseline Random Forest Model Performance:")
    evaluate_model(y_test, y_pred_rf_base)
    
    # -------------------------------
    # Hyperparameter Tuning for Random Forest
    # -------------------------------
    print("Tuning Random Forest hyperparameters with Hyperopt (parallelized)...")
    best_rf_params = tune_random_forest(X_tr, y_tr, X_val, y_val, max_evals=50)
    print("Best Random Forest hyperparameters found:", best_rf_params)
    
    print("Training tuned Random Forest model...")
    rf_tuned = RandomForestClassifier(
        n_estimators=best_rf_params['n_estimators'],
        max_depth=best_rf_params['max_depth'],
        min_samples_split=best_rf_params['min_samples_split'],
        min_samples_leaf=best_rf_params['min_samples_leaf'],
        max_features=best_rf_params['max_features'],
        random_state=42,
        n_jobs=-1
    )
    rf_tuned.fit(X_train, y_train)
    y_pred_rf_tuned = rf_tuned.predict(X_test)
    print("\nTuned Random Forest Model Performance:")
    evaluate_model(y_test, y_pred_rf_tuned)
    
    # -------------------------------
    # Hyperparameter Tuning for XGBoost
    # -------------------------------
    print("Tuning XGBoost hyperparameters with Hyperopt (parallelized)...")
    best_xgb_params = tune_xgboost(X_tr, y_tr, X_val, y_val, max_evals=50)
    print("Best XGBoost hyperparameters found:", best_xgb_params)
    
    print("Training tuned XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        **best_xgb_params,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print("\nTuned XGBoost Model Performance:")
    evaluate_model(y_test, y_pred_xgb)
    
    print("\nModel training, hyperparameter tuning, and evaluation complete.")

    # Save the tuned Random Forest model
    rf_model_path= 'models/rf_tuned_model.pkl'
    joblib.dump(rf_tuned, rf_model_path)
    print(f'Tuned Random Forest model saved to {rf_model_path}')

    # Save tuned XGBoost model
    xgb_model_path= 'models/xgb_tuned_model.pkl'
    joblib.dump(xgb_model, xgb_model_path)
    print(f'Tuned XGBoost model saved to {xgb_model_path}')
    
if __name__ == '__main__':
    main()