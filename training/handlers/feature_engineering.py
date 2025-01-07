import re

import en_core_web_sm
import pandas as pd
from helpers.text_processing import TextProcessing
from spacy.lang.en.stop_words import STOP_WORDS
from utils.logger import setup_logger

logger = setup_logger()


nlp = en_core_web_sm.load()


def add_temporal_features(df):
    logger.info("Feature engineering, add_temporal_features.")
    # Transformation of dates into years
    df["last_modified_year"] = pd.to_datetime(df["last_modified_datetime"]).dt.year
    df = df.drop(columns=["last_modified_datetime"])

    return df


def preprocessed_nutriscore_grade(df):
    logger.info("Feature engineering, preprocessed_nutriscore_grade.")
    nutriscore_grade_mapping = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "unknown": 0,
        "not-applicable": -1,
    }

    df["preprocessed_nutriscore_grade"] = df["nutriscore_grade"].replace(
        nutriscore_grade_mapping
    )
    return df


def preprocessed_ecoscore_grade(df):
    logger.info("Feature engineering, preprocessed_ecoscore_grade.")
    ecoscore_grade_mapping = {
        "a-plus": 1,
        "a": 2,
        "b": 3,
        "c": 4,
        "d": 5,
        "e": 6,
        "f": 7,
        "unknown": 0,
        "not-applicable": -1,
    }

    df["preprocessed_ecoscore_grade"] = df["ecoscore_grade"].replace(
        ecoscore_grade_mapping
    )
    return df


def clean_text_column(text):
    # Deleting language prefixes (en: - fr: ...)
    text = re.sub(r"\b\w{2}:\b", " ", text)
    # Replacing dashes with spaces
    text = text.replace("-", " ")
    # Delete multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def batch_lemmatize_text(text_series, nlp, batch_size=50):
    """
    Optimized lemmatization function for a series of texts.

    Parameters:
        text_series (pd.Series): The text series to be processed (e.g. a column in a DataFrame).
        nlp (spacy.language): Spacy model loaded for lemmatization.
        batch_size (int): The number of texts processed per batch (default 50).

    Returns:
        list: A list of lemmatized texts.
    """
    logger.info("Feature engineering, batch_lemmatize_text.")
    # Deactivate unnecessary components to speed up processing
    with nlp.select_pipes(disable=["ner", "parser"]):
        # Batch processing
        lemmatized_texts = [
            " ".join([token.lemma_ for token in doc if token.lemma_ not in STOP_WORDS])
            for doc in nlp.pipe(text_series, batch_size=batch_size)
        ]

    return lemmatized_texts


def create_preprocessed_features(df):
    logger.info("Feature engineering, create_preprocessed_features.")
    text_processing = TextProcessing()

    # Categorials features to treat
    columns_cat = [
        {"name": "packaging_tags", "lemmatize": True},
        {"name": "categories_tags", "lemmatize": True},
        {"name": "ingredients_tags", "lemmatize": True},
        {"name": "ingredients_analysis_tags", "lemmatize": True},
        {"name": "nutrient_levels_tags", "lemmatize": True},
        {"name": "main_category", "lemmatize": True},
        {"name": "popularity_tags", "lemmatize": True},
    ]

    # List of new features
    new_features_cat_names = []

    for col in columns_cat:
        # New feature name
        preprocessed_col = f"preprocessed_{col["name"]}"
        logger.info(
            f"Feature engineering, new feature name created : {preprocessed_col}."
        )

        new_columns = [preprocessed_col]

        # Clean if needed
        df[preprocessed_col] = df[col["name"]].fillna("").apply(clean_text_column)

        # Standardisation
        df[preprocessed_col] = df[preprocessed_col].apply(text_processing.standardize)

        # Apply lemmatization and delete STOP WORDS
        if col["lemmatize"] is True:
            # Create lemmatize feature
            lemmatized_col = f"preprocessed_{col["name"]}_lemmatized"
            new_columns.append(lemmatized_col)
            # Lematize
            df[lemmatized_col] = batch_lemmatize_text(df[preprocessed_col], nlp)

        # Add new column names to the list
        new_features_cat_names.extend(new_columns)
    return df
