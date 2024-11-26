import sys
import os

# todo Optimize
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
import os
import polars as pl
from training.utils.config import load_config

DTYPES = {
    "code": pl.Utf8,
    "url": pl.Utf8,
    "last_modified_t": pl.Int64,
    "product_name": pl.Utf8,
    "packaging_tags": pl.Utf8,
    "categories_tags": pl.Utf8,
    "ingredients_tags": pl.Utf8,
    "ingredients_analysis_tags": pl.Utf8,
    "allergens": pl.Utf8,
    "traces_tags": pl.Utf8,
    "additives_tags": pl.Utf8,
    "nutriscore_grade": pl.Utf8,
    "food_groups_tags": pl.Utf8,
    "states_tags": pl.Utf8,
    "ecoscore_grade": pl.Utf8,
    "nutrient_levels_tags": pl.Utf8,
    "popularity_tags": pl.Utf8,
    "main_category": pl.Utf8,
    "image_url": pl.Utf8,
    "image_small_url": pl.Utf8,
    "energy_100g": pl.Float32,
    "fat_100g": pl.Float32,
    "saturated-fat_100g": pl.Float32,
    "cholesterol_100g": pl.Float32,
    "sugars_100g": pl.Float32,
    "proteins_100g": pl.Float32,
    "salt_100g": pl.Float32,
    "fruits-vegetables-nuts-estimate-from-ingredients_100g": pl.Float32,
    "last_modified_year": pl.Int32,
    "preprocessed_nutriscore_grade": pl.Utf8,
    "preprocessed_ecoscore_grade": pl.Utf8,
    "preprocessed_product_name": pl.Utf8,
    "preprocessed_packaging_tags": pl.Utf8,
    "preprocessed_packaging_tags_lemmatized": pl.Utf8,
    "preprocessed_categories_tags": pl.Utf8,
    "preprocessed_categories_tags_lemmatized": pl.Utf8,
    "preprocessed_ingredients_tags": pl.Utf8,
    "preprocessed_ingredients_tags_lemmatized": pl.Utf8,
    "preprocessed_ingredients_analysis_tags": pl.Utf8,
    "preprocessed_ingredients_analysis_tags_lemmatized": pl.Utf8,
    "preprocessed_nutrient_levels_tags": pl.Utf8,
    "preprocessed_nutrient_levels_tags_lemmatized": pl.Utf8,
    "preprocessed_main_category": pl.Utf8,
    "preprocessed_main_category_lemmatized": pl.Utf8,
    "preprocessed_popularity_tags": pl.Utf8,
    "preprocessed_popularity_tags_lemmatized": pl.Utf8,
    "cluster_text": pl.Utf8,
}


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} - [{levelname}] - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    return logging.getLogger(__name__)


def load_and_merge_csv(file1, file2, output_file=None):
    """
    Checks that two files exist, loads them with Polars and merges them line by line.

    Args:
        file1 (str): Path to first CSV file.
        file2 (str): Path to second CSV file.
        output_file (str): Path to save output as CSV (optional).

    Returns:
        pl.DataFrame: The merged DataFrame.
    """
    if not os.path.exists(file1):
        raise FileNotFoundError(f"File '{file1}' cannot be found.")
    if not os.path.exists(file2):
        raise FileNotFoundError(f"File '{file2}' cannot be found.")

    df1 = pl.read_csv(file1, schema_overrides=DTYPES)
    df2 = pl.read_csv(file2)

    if len(df1) != len(df2):
        raise ValueError(
            "Both CSV files must have the same number of lines for a hstack."
        )

    merged_df = pl.concat([df1, df2], how="horizontal")

    if output_file:
        merged_df.write_csv(output_file)
        logger.info(f"The merged file has been saved in'{output_file}'.")

    return merged_df


logger = setup_logger()

if __name__ == "__main__":
    logger.info("🚀  Create API database script launched  🚀")
    parser = argparse.ArgumentParser(description="Starting the ML pipeline")
    parser.add_argument(
        "--config", type=str, required=True, help="Chemin du fichier de configuration"
    )
    args = parser.parse_args()
    config = load_config(args.config)

    try:
        merged_df = load_and_merge_csv(
            config["data"]["production_data_path"],
            config["data"]["training_categorical_features_data_path"],
            output_file=config["data"]["production_text_data_api_path"],
        )
        print(merged_df)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"{e}")
