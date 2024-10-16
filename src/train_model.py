"""
This code is used to train the model.
Author: Aravindh P
"""

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.load_data import load_train_data, load_val_data
from src.metrics import evaluate_model


def create_pipeline(steps: int, class_weights: str = "balanced", jobs=-1) -> Pipeline:
    """
    This function is used to create the modeling pipeline.

    Args:
        steps: It is the number of iterations for the model before convergence.
        class_weights: It is the used to balance the target classes if the data is imbalanced.
        jobs: The number of parallel process to run.

    Returns:
        The function returns the constructed pipeline.
    """
    pipe = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(),
            ),  # This is to convert text into numerical representations.
            ("scaler", StandardScaler(with_mean=False)),  # This is to scale the data.
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weights, max_iter=steps, n_jobs=jobs
                ),
            ),  # This is the model used for training on the data.
        ]
    )

    return pipe


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def train_model(config: DictConfig):
    """
    This function is configuration function used to train the model.

    Args:
        config: This is the YAML config info.
    """

    print(f"Train modeling using {config.data.processed.train}")
    print(f"Model used: {config.model.name}")
    print(f"Save the outputs to {config.data.final}\n")

    pipe = create_pipeline(config.model.parameters.steps)

    # Training Data Modeling
    train_x, train_y = load_train_data(config.data.processed)
    pipe.fit(train_x, train_y)
    y_pred = pipe.predict(train_x)
    train_metric = evaluate_model(train_y, y_pred)
    train_metric = pd.Series(train_metric, name="train_metric")

    # Val Data Prediction
    val_x, val_y = load_val_data(config.data.processed)
    y_val_pred = pipe.predict(val_x)
    val_metric = evaluate_model(val_y, y_val_pred)
    val_metric = pd.Series(val_metric, name="val_metric")

    # Metrics
    metrics = pd.concat([train_metric, val_metric], axis=1)
    metrics.to_csv(config.data.final)
    print(metrics)


if __name__ == "__main__":
    train_model()
