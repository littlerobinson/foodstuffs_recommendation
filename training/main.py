import argparse
import sys
import time
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from handlers import data_loader
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, Schema
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
    # "product_name",
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


def log_cluster_distribution_artifact(labels, n_clusters):
    """
    Generate and log the cluster distribution histogram as an artifact in MLflow using Plotly.

    Parameters:
    labels (array-like): The cluster labels.
    n_clusters (int): The number of clusters.
    """
    fig = px.histogram(
        x=labels,
        nbins=n_clusters,
        title="Cluster Distribution",
        labels={"x": "Cluster", "y": "Frequency"},
    )
    fig.write_html("artifacts/cluster_distribution.html")
    mlflow.log_artifact("artifacts/cluster_distribution.html")
    logger.info("Cluster distribution histogram logged as an artifact.")


def log_pca_cluster_plot(features, labels, product_names):
    """
    Perform PCA on the features and log the 2D cluster plot as an artifact in MLflow using Plotly.

    Parameters:
    features (DataFrame): The feature data.
    labels (array-like): The cluster labels.
    product_names (array-like): The product names.
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    df_pca = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
    df_pca["Cluster"] = labels
    df_pca["Product Name"] = product_names

    fig = px.scatter(
        df_pca,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="PCA of Clusters",
        labels={"color": "Cluster"},
        hover_data=["Product Name"],
    )
    fig.update_traces(
        marker=dict(size=5, opacity=0.7)
    )  # Adjust marker size and opacity for better visibility
    fig.write_html("artifacts/pca_cluster_plot.html")
    mlflow.log_artifact("artifacts/pca_cluster_plot.html")
    logger.info("PCA cluster plot logged as an artifact.")


def log_truncatedsvd_cluster_plot(features, labels, product_names):
    """
    Perform TruncatedSVD on the features and log the 2D cluster plot as an artifact in MLflow using Plotly.

    Parameters:
    features (DataFrame): The feature data.
    labels (array-like): The cluster labels.
    product_names (array-like): The product names.
    """
    svd = TruncatedSVD(n_components=2)
    svd_result = svd.fit_transform(features)
    df_svd = pd.DataFrame(data=svd_result, columns=["SVD1", "SVD2"])
    df_svd["Cluster"] = labels
    df_svd["Product Name"] = product_names

    fig = px.scatter(
        df_svd,
        x="SVD1",
        y="SVD2",
        color="Cluster",
        title="TruncatedSVD of Clusters",
        labels={"color": "Cluster"},
        hover_data=["Product Name"],
    )
    fig.update_traces(
        marker=dict(size=5, opacity=0.7)
    )  # Adjust marker size and opacity for better visibility
    fig.write_html("artifacts/truncatedsvd_cluster_plot.html")
    mlflow.log_artifact("artifacts/truncatedsvd_cluster_plot.html")
    logger.info("TruncatedSVD cluster plot logged as an artifact.")


def log_tsne_cluster_plot(features, labels, product_names):
    """
    Perform t-SNE on the features and log the 2D cluster plot as an artifact in MLflow using Plotly.

    Parameters:
    features (DataFrame): The feature data.
    labels (array-like): The cluster labels.
    product_names (array-like): The product names.
    """
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)
    df_tsne = pd.DataFrame(data=tsne_result, columns=["TSNE1", "TSNE2"])
    df_tsne["Cluster"] = labels
    df_tsne["Product Name"] = product_names

    fig = px.scatter(
        df_tsne,
        x="TSNE1",
        y="TSNE2",
        color="Cluster",
        title="t-SNE of Clusters",
        labels={"color": "Cluster"},
        hover_data=["Product Name"],
    )
    fig.update_traces(
        marker=dict(size=5, opacity=0.7)
    )  # Adjust marker size and opacity for better visibility
    fig.write_html("artifacts/tsne_cluster_plot.html")
    mlflow.log_artifact("artifacts/tsne_cluster_plot.html")
    logger.info("t-SNE cluster plot logged as an artifact.")


def main(config_path: str):
    # load config variables
    config = load_config(config_path)
    processed_data_path = config["data"]["processed_data_path"]
    production_data_path = config["data"]["production_data_path"]
    production_encoded_features_path = config["data"][
        "production_encoded_features_path"
    ]
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
        # processed_data = data_loader.load_dataset(processed_data_path, nrows=10000)
        processed_data = data_loader.load_dataset(processed_data_path)

    # Launch mlflow pipeline
    if args.mlflow:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)

        # Get our experiment info
        mlflow.autolog(log_models=False)
        experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
            start_time = time.time()
            model, features, metrics, labels = training_pipeline.build_pipeline(
                data=processed_data,
                n_clusters=n_clusters,
                encoding_method_name=encoding_method_name,
            )
            training_time = time.time() - start_time
            mlflow.log_metric("training_time", training_time)
            if labels is not None and len(labels) > 0:
                mlflow.log_param("labels", str(labels[:10]))
                logger.info("Save labels to data.")
                processed_data["cluster_text"] = labels
                logger.info("Save data as production database.")
                processed_data.to_csv(production_data_path, index=False)
                features.to_csv(production_encoded_features_path, index=False)

                # Log the cluster distribution histogram
                log_cluster_distribution_artifact(labels, n_clusters)

                # Log the PCA cluster plot
                log_pca_cluster_plot(features, labels, processed_data["product_name"])

                # Log the t-SNE cluster plot
                log_tsne_cluster_plot(features, labels, processed_data["product_name"])

                # Log the TruncatedSVD cluster plot
                log_truncatedsvd_cluster_plot(
                    features, labels, processed_data["product_name"]
                )
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
