import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.components.DataIngestion import DataIngestion
from src.components.DataTransformation import TransformData
from src.components.ModelTrainer import ModelTrainer
from src.logger.custom_logger import logger
from src.constants import MODEL_DIR_PATH
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("diabetes_model_tuning")


def GetBestModel():
    """Get the best model from ModelTrainer."""
    X_train, X_test, y_train, y_test = TransformData().InitiateTransformation()
    trainer = ModelTrainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    trainer.train_and_evaluate()
    return trainer.get_best_model()


def create_pipeline(trial, best_model=None):
    """Create a scikit-learn pipeline with hyperparameters suggested by Optuna."""
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    # Set parameters for RandomForestClassifier
    best_model.set_params(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

    # Create a pipeline with a scaler and the classifier
    pipeline = Pipeline([("scaler", StandardScaler()), ("rf", best_model)])
    return pipeline


def objective(trial, best_model):
    """Objective function for Optuna to minimize."""
    df = pd.read_csv("data/external/trf_df.csv")
    X = df.drop(columns=df.columns[-1])
    y = df[df.columns[-1]]

    # Create pipeline with the best model (only updating hyperparameters)
    pipeline = create_pipeline(trial, best_model)

    # Log parameters for this trial
    params = {
        "n_estimators": pipeline.named_steps["rf"].n_estimators,
        "max_depth": pipeline.named_steps["rf"].max_depth,
        "min_samples_split": pipeline.named_steps["rf"].min_samples_split,
        "min_samples_leaf": pipeline.named_steps["rf"].min_samples_leaf,
    }
    mlflow.log_params(params)

    # Use Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    mean_score = scores.mean()
    std_score = scores.std()

    # Log metrics
    mlflow.log_metrics(
        {
            "mean_cv_accuracy": mean_score,
            "std_cv_accuracy": std_score,
            "min_cv_accuracy": scores.min(),
            "max_cv_accuracy": scores.max(),
        }
    )

    return mean_score


def optimize_rf(best_model, n_trials=20, study_name="rf_optuna_study", storage_path=None):
    """Optimize RandomForestClassifier hyperparameters using Optuna."""
    with mlflow.start_run(run_name="hyperparameter_optimization"):
        if storage_path:
            storage = f"sqlite:///{storage_path}"
        else:
            storage = None

        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("study_name", study_name)

        study = optuna.create_study(
            direction="maximize", study_name=study_name, storage=storage, load_if_exists=True
        )

        # Pass the best model to the objective function
        study.optimize(lambda trial: objective(trial, best_model), n_trials=n_trials)

        # Log best trial results
        mlflow.log_params(
            {
                "best_trial_params": study.best_trial.params,
                "best_trial_value": study.best_trial.value,
            }
        )

        # Log optimization history as a JSON artifact
        history = {
            "values": [t.value for t in study.trials],
            "params": [t.params for t in study.trials],
        }
        mlflow.log_dict(history, "optimization_history.json")

        return study


def save_best_model(study, best_model, model_path=MODEL_DIR_PATH):
    """Train and save the best model found by Optuna."""
    with mlflow.start_run(run_name="save_best_model"):
        best_trial = study.best_trial
        logger.info(f"Best trial parameters: {best_trial.params}")
        logger.info(f"Best trial accuracy: {best_trial.value}")

        # Log best parameters
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_accuracy", best_trial.value)

        # Apply best parameters to the model
        params = best_trial.params
        best_model.set_params(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
        )

        # Create and train the final pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("rf", best_model)])
        df = pd.read_csv("data/external/trf_df.csv")
        X = df.drop(columns=df.columns[-1])
        y = df[df.columns[-1]]
        pipeline.fit(X, y)

        # Log the final model with signature
        signature = infer_signature(X, y)
        mlflow.sklearn.log_model(
            pipeline,
            "optimized_model",
            signature=signature,
            input_example=X.iloc[:5],
        )

        # Save the model locally
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(pipeline, model_path)
        logger.info(f"Best model saved to {model_path}")

        # Log model path as artifact
        mlflow.log_artifact(model_path, "final_model")
