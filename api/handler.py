import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


NUMERIC_COLUMNS = [
    "energy_100g",
    "fat_100g",
    "saturated-fat_100g",
    "cholesterol_100g",
    "sugars_100g",
    "proteins_100g",
    "salt_100g",
    "fruits-vegetables-nuts-estimate-from-ingredients_100g",
    "preprocessed_nutriscore_grade",
    "preprocessed_ecoscore_grade",
]
CATEGORIAL_COLUMNS = [
    "product_name",
    "packaging_tags",
    "categories_tags",
    "ingredients_tags",
    "ingredients_analysis_tags",
    "main_category",
]


def compute_cosine_similarity(target_embedding, all_embeddings):
    """
    Computes cosine similarity between a target embedding and all other embeddings.

    Parameters:
        target_embedding (np.ndarray): Embedding of the target item.
        all_embeddings (np.ndarray): Array of embeddings for other items.

    Returns:
        np.ndarray: Cosine similarity scores with the target embedding.
    """
    similarities = cosine_similarity(target_embedding.reshape(1, -1), all_embeddings)
    return similarities.flatten()


async def find_similar_products_text(code, allergen=None, top_n=10):
    """
    Finds similar products within the same cluster, avoiding those containing a specific allergen.

    Parameters:
        code (str): Code of the reference product.
        allergen (str): Allergen to avoid, if specified.
        top_n (int): Number of similar products to return.

    Returns:
        DataFrame: Similar products sorted by cosine similarity.
    """
    # Load the dataset
    df = pd.read_csv("./data/production/database.csv")

    # 1. Identify the cluster of the reference product
    product_cluster = df.loc[df["code"] == code, "cluster_text"].values[0]
    target_features = df.loc[df["code"] == code, NUMERIC_COLUMNS].values

    # 2. Load and merge categorical features
    encoded_categorical_features = pd.read_csv(
        "./data/production/categorical_features.csv"
    )
    cluster_features_combined = pd.concat([df, encoded_categorical_features], axis=1)

    # 3. Filter products within the same cluster
    similar_cluster_products = cluster_features_combined[
        cluster_features_combined["cluster_text"] == product_cluster
    ]

    # 4. Filter out products containing the specified allergen, if necessary
    if allergen:
        similar_cluster_products = similar_cluster_products[
            ~similar_cluster_products["allergens"]
            .fillna("")
            .str.contains(allergen, case=False)
            & ~similar_cluster_products["traces_tags"]
            .fillna("")
            .str.contains(allergen, case=False)
        ]

    # 5. Select only numeric columns for similarity calculation
    similar_cluster_numeric_features = similar_cluster_products[NUMERIC_COLUMNS].values

    # 6. Compute cosine similarity
    similarities = compute_cosine_similarity(
        target_features, similar_cluster_numeric_features
    )

    # 7. Add similarity scores to the filtered products
    similar_cluster_products = similar_cluster_products.assign(
        similarity_text=similarities
    )

    # 8. Sort and return the most similar products
    response = similar_cluster_products.sort_values(
        by="similarity_text", ascending=False
    )[
        [
            "code",
            "url",
            "product_name",
            "cluster_text",
            "allergens",
            "traces_tags",
            "image_url",
            "nutriscore_grade",
            "ecoscore_grade",
            "similarity_text",
        ]
    ].head(top_n)

    return response.T.to_json()


def find_similar_products_img(
    product_code,
    top_n,
):
    """
    Trouve les produits similaires dans le même cluster basé sur la similarité cosinus, en utilisant le code du produit.
    Le DataFrame retourné contient uniquement les produits du même cluster et une colonne 'similarity_img'
    avec les scores de similarité, triés par ordre décroissant.

    Parameters:
        product_code (str/int): Le code du produit pour lequel trouver des produits similaires.
        df_path (str): Chemin vers le fichier CSV contenant les informations sur les produits.
        product_code_column (str): Nom de la colonne contenant les codes des produits.
        embedding_prefix (str): Préfixe des colonnes contenant les dimensions des embeddings des produits.
        cluster_column (str): Nom de la colonne contenant les clusters des produits.
        top_n (int): Nombre de produits similaires à retourner.

    Returns:
        str: JSON contenant les produits similaires.
    """
    product_code_column="code"
    embedding_prefix="embedding_"
    cluster_column="cluster_emb"

    # Charger le DataFrame
    df = pd.read_csv("./data/clean_dataset_clusters.csv")

    print(df[[product_code_column]].dtypes)

    # Vérifier que les colonnes nécessaires existent
    embedding_columns = [col for col in df.columns if col.startswith(embedding_prefix)]
    # print(embedding_columns)
    if not embedding_columns:
        raise KeyError(
            f"Les colonnes avec le préfixe '{embedding_prefix}' doivent exister dans le DataFrame."
        )

    # Trouver l'entrée pour le produit donné
    target_row = df[df[product_code_column] == product_code]
    if target_row.empty:
        raise ValueError(
            f"Le produit avec le code '{product_code}' n'existe pas dans le DataFrame."
        )

    # Obtenir l'embedding et le cluster du produit cible
    target_embedding = target_row[embedding_columns].values
    if target_embedding.shape[0] == 0 or np.isnan(target_embedding).any():
        raise ValueError(
            f"L'embedding du produit avec le code '{product_code}' est vide ou invalide."
        )
    target_cluster = target_row.iloc[0][cluster_column]

    # Filtrer les produits appartenant au même cluster
    cluster_products = df[df[cluster_column] == target_cluster]

    # Supprimer la ligne du produit cible du cluster pour éviter la comparaison avec lui-même
    cluster_products = cluster_products[
        cluster_products[product_code_column] != product_code
    ]

    # Calculer les similarités cosinus avec les produits du cluster
    cluster_embeddings = cluster_products[embedding_columns].values
    similarities = cosine_similarity(target_embedding, cluster_embeddings).flatten()

    # Ajouter la colonne 'similarity_img' au DataFrame des produits du même cluster
    cluster_products = cluster_products.copy()
    cluster_products["similarity_img"] = similarities

    # Trier par similarité et garder les top_n
    cluster_products = cluster_products.sort_values(
        by="similarity_img", ascending=False
    )
    response = cluster_products.drop(columns=embedding_columns).head(top_n)

    return response.T.to_json()
