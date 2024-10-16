"""
This is code is used to Load the dataset into memory.

Author: Aravindh P
"""

import pandas as pd


def load_train_data(paths: dict) -> tuple[pd.Series, pd.Series]:
    """
    This functions loads the training data into memory.

    Args:
        paths: This contains the paths of the datasets.

    Returns:
        They data is finally returned as dependent features and target variable.
    """
    df = pd.read_parquet(paths["train"])
    x, y = df.sentence, df.label
    return x, y


def load_val_data(paths: dict) -> tuple[pd.Series, pd.Series]:
    """
    This functions loads the validation data into memory.

    Args:
        paths: This contains the paths of the datasets.

    Returns:
        They data is finally returned as dependent features and target variable.
    """
    df = pd.read_parquet(paths["val"])
    x, y = df.sentence, df.label
    return x, y
