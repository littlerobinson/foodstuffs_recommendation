from utils.config import load_config
from handlers.download_image import download_all_images
from utils.logger import setup_logger
from pipelines.images.training_pipeline import perform_clustering
import mlflow
import numpy as np
import argparse


logger = setup_logger()

from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np


def get_signature(df_path, embedding_column):
    """
    Crée la signature des données d'entrée et de sortie pour le modèle.

    Parameters:
        df_path (str): Le chemin vers le fichier pickle contenant les données.
        embedding_column (str): Le nom de la colonne contenant les embeddings.

    Returns:
        ModelSignature: La signature des données d'entrée.
    """
    # Charger le DataFrame
    df = pd.read_pickle(df_path)

    # Récupérer les embeddings valides
    valid_embeddings_df = df[df[embedding_column].notnull()].copy()
    embeddings = valid_embeddings_df[embedding_column].tolist()

    # Convertir les embeddings en un tableau NumPy 2D
    valid_embeddings = np.array([e for e in embeddings if len(e) == len(embeddings[0])])

    if valid_embeddings.ndim != 2:
        raise ValueError("Les embeddings doivent être une matrice 2D.")

    # Construire un DataFrame d'exemple pour la signature d'entrée
    input_example = pd.DataFrame(
        valid_embeddings, columns=[f"dim_{i}" for i in range(valid_embeddings.shape[1])]
    )

    # Inférer la signature à partir de l'exemple
    signature = infer_signature(input_example)

    return signature


def main(config_path: str):
    # Load configuration from file
    config = load_config(config_path)

    df_path = config["data"]["clean_data_with_embed"]
    embedding_column = config["data"]["embedding_column"]
    save_df_path = config["data"]["data_with_clusters"]
    n_clusters = config["image_training"]["n_clusters"]
    mlflow_experiment_name = config["training"]["mlflow_experiment_name"]
    mlflow_tracking_uri = config["training"]["mlflow_tracking_uri"]
    mlflow_model_name = config["training"]["mlflow_model_name"]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    mlflow.autolog(log_models=False)
    experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        model, metrics, labels = perform_clustering(
            df_path, embedding_column, n_clusters, save_df_path
        )
    for metric in metrics:
        mlflow.log_metric(metric, metrics[metric])
    cluster_count = np.unique(labels).size
    mlflow.log_metric("cluster_count", cluster_count)

    # Generate signature
    mlflow_signature = get_signature(df_path, embedding_column)

    # Log the sklearn model and register as version
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=mlflow_experiment_name,
        registered_model_name=mlflow_model_name,
        signature=mlflow_signature,
    )

if __name__ == "__main__":
    logger.info("🚀  Clustering starting 🚀")
    
    parser = argparse.ArgumentParser(description="Starting the ML pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Chemin du fichier de configuration"
    )
    args = parser.parse_args()
    main(args.config)
