from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

def perform_clustering(df_path, embedding_column, n_clusters, save_df_path):
    """
    Applique l'algorithme de clustering KMeans sur une colonne d'embeddings d'un DataFrame.

    Parameters:
        df_path (str): Le chemin du DataFrame contenant les embeddings.
        embedding_column (str): Le nom de la colonne contenant les embeddings.
        n_clusters (int): Le nombre de clusters optimal déterminé.
        save_df_path (str): Le chemin du DataFrame final avec la colonne 'cluster' ajoutée.

    Returns:
        DataFrame: Le DataFrame d'origine avec une colonne supplémentaire 'cluster' contenant les labels des clusters.
    """
    # Load dataset
    df = pd.read_pickle(df_path)

    # Filtrer les lignes avec des embeddings valides
    valid_embeddings_df = df[df[embedding_column].notnull()].copy()
    embeddings = valid_embeddings_df[embedding_column].tolist()

    # Vérifier la consistance des embeddings (même dimension pour tous)
    try:
        valid_embeddings = np.array(
            [e for e in embeddings if len(e) == len(embeddings[0])]
        )
    except IndexError:
        raise ValueError(
            "Les embeddings contiennent des valeurs invalides ou inconsistantes."
        )

    if len(valid_embeddings) != len(embeddings):
        print(
            f"Warning: {len(embeddings) - len(valid_embeddings)} invalid embeddings were removed."
        )

    # Vérifier que les embeddings sont 2D
    if valid_embeddings.ndim != 2:
        raise ValueError(
            "Les embeddings doivent être une matrice 2D (chaque ligne étant un vecteur d'embedding)."
        )

    # Appliquer le clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(valid_embeddings)

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

    # Ajouter les labels des clusters au DataFrame
    valid_embeddings_df["cluster_emb"] = cluster_labels
    valid_embeddings_df["cluster_emb"] = valid_embeddings_df["cluster_emb"].astype(int)

    # Vérifier si "cluster_emb" existe dans df si oui le supprimer
    if "cluster_emb" in df.columns:
        df = df.drop(columns=["cluster_emb"])

    # Fusionner les résultats avec le DataFrame original
    df = df.merge(
        valid_embeddings_df[["cluster_emb"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    df.to_csv(save_df_path, index=False)
    return kmeans, metrics, cluster_labels
