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
    Applique l'algorithme de clustering KMeans sur des colonnes d'embeddings éclatées dans un DataFrame.

    Parameters:
        df_path (str): Le chemin du DataFrame contenant les embeddings.
        embedding_prefix (str): Le préfixe des colonnes contenant les dimensions des embeddings.
        n_clusters (int): Le nombre de clusters optimal déterminé.
        save_df_path (str): Le chemin du DataFrame final avec la colonne 'cluster' ajoutée.

    Returns:
        Tuple: KMeans object, metrics dictionary, cluster labels.
    """
    # Charger le DataFrame
    df = pd.read_csv(df_path)

    # Filtrer les colonnes correspondant aux dimensions des embeddings
    embedding_columns = [col for col in df.columns if col.startswith(embedding_prefix)]

    if not embedding_columns:
        raise ValueError(
            f"Aucune colonne trouvée avec le préfixe '{embedding_prefix}'."
        )

    # Filtrer les lignes avec des valeurs valides pour toutes les dimensions
    valid_embeddings_df = df.dropna(subset=embedding_columns)

    # Extraire les embeddings sous forme de matrice NumPy
    valid_embeddings = valid_embeddings_df[embedding_columns].values

    # Vérifier que les embeddings sont 2D
    if valid_embeddings.ndim != 2:
        raise ValueError(
            "Les embeddings doivent être une matrice 2D (chaque ligne étant un vecteur d'embedding)."
        )

    # Appliquer le clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(valid_embeddings)

    # Calcul des métriques
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
    valid_embeddings_df = valid_embeddings_df.copy()
    valid_embeddings_df["cluster_emb"] = cluster_labels

    # Convertir en entier si nécessaire
    valid_embeddings_df["cluster_emb"] = valid_embeddings_df["cluster_emb"].astype(int)

    # Vérifier si "cluster_emb" existe dans df, le supprimer si nécessaire
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