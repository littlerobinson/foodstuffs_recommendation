from utils.logger import setup_logger

from pipelines import data_cleaning, data_loader, feature_engineering

logger = setup_logger()


def build_pipeline(raw_data_path, processed_data_path):
    logger.info("Build data pipeline, work in progress 🚧.")
    # Step 1: Load data
    rawdata = data_loader.extract_raw_data(raw_data_path)

    # Step 2: Clean data
    df = data_cleaning.clean_dataset(rawdata)
    df = data_cleaning.remove_duplicates(df)

    # Step 3: Feature Engineering
    df = feature_engineering.add_temporal_features(df)
    df = feature_engineering.preprocessed_nutriscore_grade(df)
    df = feature_engineering.preprocessed_ecoscore_grade(df)
    df = feature_engineering.create_preprocessed_features(df)

    # Save preprocessed dataset as csv for training
    df.to_csv(processed_data_path)

    logger.info("Build data pipeline, finish 🎉.")
