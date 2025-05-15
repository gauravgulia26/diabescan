import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.components.DataIngestion import DataIngestion
from src.components.DataTransformation import TransformData
from src.components.ModelTrainer import ModelTrainer
from src.logger.custom_logger import logger
from src.constants import MODEL_DIR_PATH
import numpy as np
import pandas as pd
import joblib
import os

def GetBestModel():
    """Get the best model from ModelTrainer."""
    train_df, test_df = TransformData().InitiateTransformation()
    X_train = train_df.drop(columns=train_df.columns[-1])
    y_train = train_df[train_df.columns[-1]]
    X_test = test_df.drop(columns=train_df.columns[-1])
    y_test = test_df[train_df.columns[-1]]
    trainer = ModelTrainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    trainer.train_and_evaluate()
    return trainer.get_best_model()


def create_pipeline(trial, best_model=None):
    """Create a scikit-learn pipeline with hyperparameters suggested by Optuna."""
    # Suggest hyperparameters
    C = trial.suggest_float("C", 1e-5, 1e5, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    
    # Handle degree parameter for poly kernel
    if kernel == "poly":
        degree = trial.suggest_int("degree", 2, 5)
        best_model.set_params(C=C, kernel=kernel, gamma=gamma, degree=degree)
    else:
        best_model.set_params(C=C, kernel=kernel, gamma=gamma)

    # Create a pipeline with a scaler and the classifier
    pipeline = Pipeline([("scaler", StandardScaler()), ("svc", best_model)])
    return pipeline


def objective(trial, best_model):
    """Objective function for Optuna to minimize."""
    df = pd.read_csv("data/processed/train/train.csv")
    X = df.drop(columns=df.columns[-1])
    y = df[df.columns[-1]]
    
    # Create pipeline with the best model (only updating hyperparameters)
    pipeline = create_pipeline(trial, best_model)

    # Use Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    return scores.mean()


def optimize_svm(best_model, n_trials=50, study_name="svm_optuna_study", storage_path=None):
    """Optimize SVM hyperparameters using Optuna."""
    if storage_path:
        storage = f"sqlite:///{storage_path}"
    else:
        storage = None

    study = optuna.create_study(
        direction="maximize", study_name=study_name, storage=storage, load_if_exists=True
    )
    
    # Pass the best model to the objective function
    study.optimize(lambda trial: objective(trial, best_model), n_trials=n_trials)
    return study

def save_best_model(study, best_model, model_path=MODEL_DIR_PATH):
    """Train and save the best model found by Optuna."""
    best_trial = study.best_trial
    logger.info(f"Best trial parameters: {best_trial.params}")
    logger.info(f"Best trial accuracy: {best_trial.value}")

    # Apply best parameters to the model
    params = best_trial.params
    
    # Set parameters based on kernel type
    if params["kernel"] == "poly":
        best_model.set_params(
            C=params["C"], 
            kernel=params["kernel"], 
            gamma=params["gamma"], 
            degree=params["degree"]
        )
    else:
        best_model.set_params(
            C=params["C"], 
            kernel=params["kernel"], 
            gamma=params["gamma"]
        )

    # Create the final pipeline
    pipeline = Pipeline([("scaler", StandardScaler()), ("svc", best_model)])

    # Train the pipeline on the entire dataset
    df = pd.read_csv("data/processed/train/train.csv")
    X = df.drop(columns=df.columns[-1])
    y = df[df.columns[-1]]
    pipeline.fit(X, y)

    # Save the trained pipeline
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info(f"Best model saved to {model_path}")