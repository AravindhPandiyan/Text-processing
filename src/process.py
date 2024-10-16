"""
This is code is used to process the dataset.

Author: Aravindh P
"""

import multiprocessing as mp
import re

import hydra
import pandas as pd
import spacy
from omegaconf import DictConfig

# Spacy small english model is loaded into memory.
nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "parser", "ner"])


def process_text(path: str, col: str, batch: int) -> pd.DataFrame:
    """
    This function is used to process all the sentences in the given data series, by removing stopwords and lemmatization of the words.

    Args:
        path: This is the path to the given data series to be processed.
        col: The column that contains the text data.
        batch: The number of sentences to be processed parallely.

    Returns:
        It returns the processed data series data-frame.
    """
    # The data series is loaded to the memory.
    df = pd.read_parquet(path)
    df = df.set_index("idx")

    # Special characters and numbers are removed, then it is converted to lower case.
    df[col] = df[col].map(lambda sen: re.sub("[^a-zA-Z]", " ", sen).lower())

    docs = nlp.pipe(
        df[col], batch_size=batch, n_process=mp.cpu_count()
    )  # The data series processing is set.

    corpus = []

    # The data series sentences are processed individually.
    for doc in docs:
        phrases = []

        for token in doc:
            if not token.is_stop:
                phrases.append(token.lemma_)

        phrase = " ".join(phrases)
        phrase = re.sub(r"\s+", " ", phrase)
        phrase = phrase.strip()
        corpus.append(phrase)

    df[col] = corpus

    return df


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def process_data(config: DictConfig):
    """
    This is the configuration function used process the raw data.

    Args:
        config: This is the YAML config info.
    """

    print(f"Process data using {config.data.raw}")
    print(f"Columns used: {config.process.use_column}")

    # Each Data Series to be processed and saved is iterated.
    for data in config.data.raw.keys():
        df = process_text(
            config.data.raw[data], config.process.use_column, config.process.batch_size
        )
        df.to_parquet(config.data.processed[data])


if __name__ == "__main__":
    process_data()
