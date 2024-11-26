from utils.logger import setup_logger

from handlers.data_params import COLUMNS_TO_KEEP

logger = setup_logger()


def clean_dataset(df):
    logger.info("Data cleaning, clean dataset.")
    # Select columns to keep
    clean_data = df[COLUMNS_TO_KEEP]

    # Delete empty rows for certain columns
    clean_data = clean_data.dropna(subset=["product_name", "ingredients_tags"])

    # Replace empty values in other columns
    fill_values = {
        "packaging_tags": "en:unknown",
        "categories_tags": "en:unknown",
        "ingredients_analysis_tags": "unknown",
        "allergens": "en:none",
        "traces_tags": "en:none",
        "additives_tags": "en:none",
        "nutriscore_grade": "unknown",
        "food_groups_tags": "en:none",
        "states_tags": "en:unknown",
        "ecoscore_grade": "unknown",
        "nutrient_levels_tags": "en:unknown",
        "popularity_tags": "unknown",
        "main_category": "en:none",
        "image_url": "en:none",
        "energy_100g": -1,
        "fat_100g": -1,
        "saturated-fat_100g": -1,
        "cholesterol_100g": -1,
        "sugars_100g": -1,
        "proteins_100g": -1,
        "salt_100g": -1,
        "fruits-vegetables-nuts-estimate-from-ingredients_100g": -1,
    }

    clean_data.fillna(value=fill_values, inplace=True)

    return clean_data


def remove_duplicates(df):
    logger.info("Data cleaning, remove duplicates data - code.")
    df_code_sorted = df.sort_values(
        by=["code", "last_modified_t"], ascending=[True, False]
    )
    # Delete duplicates, retaining the last (most recent) occurrence
    df = df_code_sorted.drop_duplicates(subset="code", keep="last")

    logger.info("Data cleaning, remove duplicates data - url.")
    df_url_sorted = df.sort_values(
        by=["url", "last_modified_t"], ascending=[True, False]
    )
    # Delete duplicates, retaining the last (most recent) occurrence
    df = df_url_sorted.drop_duplicates(subset="url", keep="last")
    return df
