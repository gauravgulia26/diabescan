from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validate_arguments
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.components.DataTransformation import TransformData
from src.constants import RESULT_DIR_PATH, MODEL_DIR_PATH
from src.logger.custom_logger import logger
import warnings
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Optional: Include XGBoost if available
try:
    from xgboost import XGBClassifier

    xgb_installed = True
except ImportError:
    xgb_installed = False


# Define a Pydantic model for configuration
class ModelTrainerConfig(BaseModel):
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)
    scoring_metric: str = Field(default="f1_score")
    scale_data: bool = Field(default=True)

    @validate_arguments
    def validate_metric(cls, v):
        allowed_metrics = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
        if v.lower() not in allowed_metrics:
            raise ValueError(f"scoring_metric must be one of {allowed_metrics}")
        return v.lower()


# Main class for training and evaluating models
class ModelTrainer:
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        config: Optional[ModelTrainerConfig] = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config or ModelTrainerConfig()
        self.models: Dict[str, Any] = {}
        self.results: List[Dict[str, Any]] = []
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self._initialize_models()

    def _initialize_models(self):
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
        }
        if xgb_installed:
            self.models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    def train_and_evaluate(self):
        logger.info("Starting Training !!")
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_proba = (
                model.predict_proba(self.X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else y_pred
            )

            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(self.y_test, y_pred),
                "Precision": precision_score(self.y_test, y_pred),
                "Recall": recall_score(self.y_test, y_pred),
                "F1 Score": f1_score(self.y_test, y_pred),
                "ROC AUC": roc_auc_score(self.y_test, y_proba),
            }
            self.results.append(metrics)

        self._select_best_model()

    def _select_best_model(self):
        df_results = pd.DataFrame(self.results)
        metric_column = self.config.scoring_metric.replace("_", " ").title()
        df_sorted = df_results.sort_values(by=metric_column, ascending=False).reset_index(
            drop=True
        )
        self.best_model_name = df_sorted.loc[0, "Model"]
        self.best_model = self.models[self.best_model_name]
        self.results_df = df_sorted

        logger.info(f"Training Completed, Best Model is {self.get_best_model_name()}")
        self.get_results()
        logger.info('Procedding to Tune Best Model')
        # logger.info("Saving the Model in Disk !!")
        # self._save_best_model()

    def get_results(self):
        results = pd.DataFrame(self.results)
        try:
            results.to_csv(os.path.join(RESULT_DIR_PATH, "results.csv"), index=False)
        except Exception as e:
            logger.error(e)
        else:
            logger.info(f"Training Results Saved at {RESULT_DIR_PATH}")

    def get_best_model(self) -> Any:
        return self.best_model

    def get_best_model_name(self) -> str:
        return str(self.best_model_name)

    # def _save_best_model(self):
    #     try:
    #         with open(MODEL_DIR_PATH, "wb") as f:
    #             pickle.dump(self.best_model, f)
    #         logger.info(f"✅ Best model '{self.best_model_name}' saved to {MODEL_DIR_PATH}")
    #     except Exception as e:
    #         logger.error(f"❌ Failed to save the model: {e}")
