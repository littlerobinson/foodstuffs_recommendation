import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from io import BytesIO
from training.utils.config import load_config
from training.utils.logger import setup_logger
from training.pipelines.images.training_pipeline import perform_clustering
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile


logger = setup_logger()


def combined_plot_to_mlflow(cluster_range, inertia, silhouette_scores):
    """
    Generate a combined Elbow and Silhouette plot, and log it as an MLflow artifact.
    """
    # Create a figure with two subplots
    fig = make_subplots(
        rows=2,
        cols=1,  # Two rows, one column
        subplot_titles=("Elbow Method (Inertia)", "Silhouette Score"),
        vertical_spacing=0.2,  # Vertical spacing between subplots
    )

    # Add the Elbow Method (Inertia) plot to the first subplot
    fig.add_trace(
        go.Scatter(
            x=cluster_range,
            y=inertia,
            mode="lines+markers",
            name="Inertia",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Add the Silhouette Score plot to the second subplot
    fig.add_trace(
        go.Scatter(
            x=cluster_range,
            y=silhouette_scores,
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="green"),
        ),
        row=2,
        col=1,
    )

    # Update layout, including dimensions and titles
    fig.update_layout(
        height=800,  # Figure height
        width=800,  # Figure width
        title_text="Clustering Metrics: Elbow Method & Silhouette Score",
    )
    fig.update_xaxes(title_text="Number of clusters", row=1, col=1)
    fig.update_xaxes(title_text="Number of clusters", row=2, col=1)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=2, col=1)

    # Save the combined plot to MLflow as an artifact
    # Save the combined plot to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        fig.write_html(tmp_file.name)
        tmp_file.seek(0)
        mlflow.log_artifact(tmp_file.name, artifact_path="combined_plot.html")


def main(config_path: str):
    config = load_config(config_path)
    df_path = config["data"]["production_data_path"]
    embedding_prefix = config["data"]["embedding_prefix"]
    save_df_path = config["data"]["production_image_data_api_path"]
    mlflow_experiment_name = config["training"]["mlflow_experiment_name"]
    mlflow_tracking_uri = config["training"]["mlflow_tracking_uri"]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.autolog(log_models=False)

    cluster_range = list(range(50, 100, 25))
    inertia_values = []
    silhouette_scores = []

    experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        for n_clusters in cluster_range:
            logger.info(f"Clustering with {n_clusters} clusters...")
            _, metrics, _ = perform_clustering(
                df_path, embedding_prefix, n_clusters, save_df_path
            )

            # Logging metrics
            for metric in metrics:
                mlflow.log_metric(f"{metric}_{n_clusters}", metrics[metric])

            # Append inertia and silhouette score for plotting
            inertia_values.append(metrics.get("kmeans_inertia", None))
            silhouette_scores.append(metrics.get("silhouette_metric", None))

            print(inertia_values, "\n", silhouette_scores)

        # Generate and log Elbow and Silhouette plots directly to MLflow
        combined_plot_to_mlflow(cluster_range, inertia_values, silhouette_scores)


if __name__ == "__main__":
    logger.info("🚀  Clustering with varying number of clusters 🚀")
    import argparse

    parser = argparse.ArgumentParser(
        description="Starting the ML pipeline with varying clusters"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()
    main(args.config)
