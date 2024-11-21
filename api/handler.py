import pandas as pd
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
