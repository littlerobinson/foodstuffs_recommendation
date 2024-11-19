from handlers import data_loader
from utils.logger import setup_logger

from handlers import data_cleaning, feature_engineering
from handlers.data_params import COLUMNS_TO_KEEP

logger = setup_logger()


def build_pipeline(config):
    raw_data_path = config["data"]["raw_data_path"]
    processed_data_path = config["data"]["processed_data_path"]
    clean_data_path = config["data"]["clean_data_path"]
    production_data_path = config["data"]["production_data_path"]

    logger.info("Build data pipeline, work in progress 🚧.")

    # Step 1: Load data
    logger.info("Build data pipeline, load raw data.")
    rawdata = data_loader.extract_raw_data(raw_data_path)

    # Step 2: Clean data
    logger.info("Build data pipeline, clean data.")
    clean_data = data_cleaning.clean_dataset(rawdata)
    clean_data = data_cleaning.remove_duplicates(clean_data)

    # Step 3: Feature Engineering
    logger.info("Build data pipeline, feature engineering.")
    processed_data = feature_engineering.add_temporal_features(clean_data)
    processed_data = feature_engineering.preprocessed_nutriscore_grade(processed_data)
    processed_data = feature_engineering.preprocessed_ecoscore_grade(processed_data)
    processed_data = feature_engineering.create_preprocessed_features(processed_data)

    # Step 4: Save datas
    logger.info("Build data pipeline, save processed data.")
    processed_data.to_csv(processed_data_path, index=False)
    logger.info("Build data pipeline, save clean data.")
    clean_data.to_csv(clean_data_path, index=False)
    logger.info("Build data pipeline, save production data.")
    clean_data[COLUMNS_TO_KEEP].to_csv(production_data_path, index=False)

    logger.info("Data pipeline, finish 🎉.")
    return processed_data
