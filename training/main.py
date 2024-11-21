import argparse
import os
import sys

import mlflow
import numpy as np
from handlers import data_loader
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
from pipelines import data_pipeline, training_pipeline
from utils.config import load_config
from utils.logger import setup_logger

logger = setup_logger()


def exit_program():
    print("Exiting the program...")
    sys.exit(0)


NUMERIC_COLUMNS = [
    "energy_100g",
    "fat_100g",
    "saturated-fat_100g",
    "cholesterol_100g",
    "sugars_100g",
    "proteins_100g",
    "salt_100g",
    "fruits-vegetables-nuts-estimate-from-ingredients_100g",
    "nutriscore_grade",
    "ecoscore_grade",
]
CATEGORIAL_COLUMNS = [
    "product_name",
    "packaging_tags",
    "categories_tags",
    "ingredients_tags",
    "ingredients_analysis_tags",
    "main_category",
]


def get_mlflow_signature():
    input_schema = Schema(
        [
            ColSpec("string", "product_name"),
            ColSpec("string", "packaging_tags"),
            ColSpec("string", "categories_tags"),
            ColSpec("string", "ingredients_tags"),
            ColSpec("string", "ingredients_analysis_tags"),
            ColSpec("string", "allergens"),
            ColSpec("string", "traces_tags"),
            ColSpec("string", "additives_tags"),
            ColSpec("string", "nutriscore_grade"),
            ColSpec("string", "food_groups_tags"),
            ColSpec("string", "states_tags"),
            ColSpec("string", "ecoscore_grade"),
            ColSpec("string", "nutrient_levels_tags"),
            ColSpec("string", "popularity_tags"),
            ColSpec("string", "main_category"),
            ColSpec("string", "image_url"),
            ColSpec("string", "image_small_url"),
            ColSpec("float", "energy_100g"),
            ColSpec("float", "fat_100g"),
            ColSpec("float", "saturated-fat_100g"),
            ColSpec("float", "cholesterol_100g"),
            ColSpec("float", "sugars_100g"),
            ColSpec("float", "proteins_100g"),
            ColSpec("float", "salt_100g"),
            ColSpec("float", "fruits-vegetables-nuts-estimate-from-ingredients_100g"),
        ]
    )

    signature = ModelSignature(inputs=input_schema)

    return signature


def main(config_path: str):
    # load config variables
    config = load_config(config_path)
    processed_data_path = config["data"]["processed_data_path"]
    production_data_path = config["data"]["production_data_path"]
    n_clusters = config["training"]["n_clusters"]
    encoding_method_name = config["training"]["encoding_method_name"]
    mlflow_experiment_name = config["training"]["mlflow_experiment_name"]
    mlflow_tracking_uri = config["training"]["mlflow_tracking_uri"]
    mlflow_model_name = config["training"]["mlflow_model_name"]

    # Launch data pipeline (load, clean, preprocess, and save)
    if args.load_data:
        processed_data = data_pipeline.build_pipeline(config)
        if processed_data.empty:
            logger.warning("Raw data is empty. Check the file path or filters.")
            exit_program()
        else:
            logger.info(
                f"Data loaded and preprocessed successfully: {len(processed_data)} rows available."
            )
    else:
        processed_data = data_loader.load_dataset(processed_data_path)

    # Launch mlflow pipeline
    if args.mlflow:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)

        # Get our experiment info
        mlflow.autolog(log_models=False)
        experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
            model, metrics, labels = training_pipeline.build_pipeline(
                data=processed_data,
                n_clusters=n_clusters,
                encoding_method_name=encoding_method_name,
            )
            if labels is not None and len(labels) > 0:
                mlflow.log_param("labels", str(labels[:10]))
                logger.info("Save labels to data.")
                processed_data["cluster_text"] = labels
                logger.info("Save data as production database.")
                processed_data.to_csv(production_data_path, index=False)
            else:
                logger.warning(
                    "No labels provided for training, metrics will not be logged."
                )
            mlflow_signature = get_mlflow_signature()
            # mlflow.log_param("n_clusters", n_clusters)
            # mlflow.log_param("encoding_method", encoding_method_name)
            # logger.info(f"Log to MLFlow, experiment_name: {mlflow_experiment_name}")
            # logger.info(f"Log to MLFlow, mlflow_tracking_uri: {mlflow_tracking_uri}")
            # logger.info(f"Log dataset_path: {processed_data_path}")
            # logger.info(f"Log n_clusters: {n_clusters}")
            # logger.info(f"Log model: {model}")

            for metric in metrics:
                mlflow.log_metric(metric, metrics[metric])
            cluster_count = np.unique(labels).size
            mlflow.log_metric("cluster_count", cluster_count)
            # Log the sklearn model and register as version
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=mlflow_experiment_name,
                registered_model_name=mlflow_model_name,
                signature=mlflow_signature,
            )


if __name__ == "__main__":
    logger.info("🚀  Foodstuffs Recommendation Launched. 🚀")

    parser = argparse.ArgumentParser(description="Starting the ML pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Set the config file path",
    )
    parser.add_argument(
        "--load_data",
        default=False,
        action="store_true",
        help="Run data pipeline from raw data",
    )
    parser.add_argument(
        "--mlflow", default=False, action="store_true", help="Run MLflow"
    )

    args = parser.parse_args()
    main(args.config)
