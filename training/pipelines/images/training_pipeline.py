from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def perform_clustering(df_path, embedding_prefix, n_clusters, save_df_path):
    """
    Applies the KMeans clustering algorithm on exploded embedding columns in a DataFrame.

    Parameters:
        df_path (str): The path to the DataFrame containing the embeddings.
        embedding_prefix (str): The prefix of the columns containing the embedding dimensions.
        n_clusters (int): The optimal number of clusters determined.
        save_df_path (str): The path to the final DataFrame with the added 'cluster' column.

    Returns:
        Tuple: KMeans object, metrics dictionary, cluster labels.
    """
    # Load the DataFrame
    df = pd.read_csv(df_path)

    # Filter columns corresponding to the embedding dimensions
    embedding_columns = [col for col in df.columns if col.startswith(embedding_prefix)]

    if not embedding_columns:
        raise ValueError(f"No columns found with the prefix '{embedding_prefix}'.")

    # Filter rows with valid values for all dimensions
    valid_embeddings_df = df.dropna(subset=embedding_columns)

    # Extract embeddings as a NumPy matrix
    valid_embeddings = valid_embeddings_df[embedding_columns].values

    # Ensure the embeddings are 2D
    if valid_embeddings.ndim != 2:
        raise ValueError(
            "The embeddings must be a 2D matrix (each line being an embedding vector).s"
        )

    # Apply clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(valid_embeddings)

    # Apply clustering
    inertia = kmeans.inertia_
    silhouette_metric = silhouette_score(valid_embeddings, cluster_labels)
    davies_bouldin_metric = davies_bouldin_score(valid_embeddings, cluster_labels)
    calinski_harabasz_metric = calinski_harabasz_score(valid_embeddings, cluster_labels)

    metrics = {
        "kmeans_inertia": inertia,
        "silhouette_metric": silhouette_metric,
        "davies_bouldin_metric": davies_bouldin_metric,
        "calinski_harabasz_metric": calinski_harabasz_metric,
    }

    # Add cluster labels to the DataFrame
    valid_embeddings_df = valid_embeddings_df.copy()
    valid_embeddings_df["cluster_emb"] = cluster_labels

    # Convert to integer if necessary
    valid_embeddings_df["cluster_emb"] = valid_embeddings_df["cluster_emb"].astype(int)

    # Check if "cluster_emb" exists in the original DataFrame, remove it if necessary
    if "cluster_emb" in df.columns:
        df = df.drop(columns=["cluster_emb"])

    # Merge results with the original DataFrame
    df = df.merge(
        valid_embeddings_df[["cluster_emb"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    df.to_csv(save_df_path, index=False)

    return kmeans, metrics, cluster_labels
