from utils.config import load_config
from utils.logger import setup_logger
from pipelines.images.training_pipeline import perform_clustering
import mlflow
import numpy as np
import argparse

logger = setup_logger()

from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np


def get_signature(df_path, embedding_prefix):
    """
    Crée la signature des données d'entrée et de sortie pour le modèle.

    Parameters:
        df_path (str): Le chemin vers le fichier pickle contenant les données.
        embedding_prefix (str): Le préfixe des colonnes contenant les dimensions des embeddings.

    Returns:
        ModelSignature: La signature des données d'entrée.
    """
    # Charger le DataFrame
    df = pd.read_csv(df_path)

    # Filtrer les colonnes correspondant aux dimensions des embeddings
    embedding_columns = [col for col in df.columns if col.startswith(embedding_prefix)]

    if not embedding_columns:
        raise ValueError(
            f"Aucune colonne trouvée avec le préfixe '{embedding_prefix}'."
        )

    # Vérifier que les colonnes d'embedding ne contiennent pas de valeurs manquantes
    valid_embeddings_df = df[embedding_columns].dropna()

    # Construire un exemple d'entrée pour la signature
    input_example = valid_embeddings_df.head(1)  # Utiliser une seule ligne pour l'exemple

    # Inférer la signature à partir de l'exemple
    signature = infer_signature(input_example)

    return signature


def main(config_path: str):
    # Load configuration from file
    config = load_config(config_path)

    df_path = config["data"]["clean_data_with_embed"]
    embedding_prefix = config["data"]["embedding_prefix"]
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
            df_path, embedding_prefix, n_clusters, save_df_path
        )
        for metric in metrics:
            mlflow.log_metric(metric, metrics[metric])
        cluster_count = np.unique(labels).size
        mlflow.log_metric("cluster_count", cluster_count)

        # Generate signature
        mlflow_signature = get_signature(df_path, embedding_prefix)

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
