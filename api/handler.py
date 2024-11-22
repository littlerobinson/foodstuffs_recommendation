import pandas as pd
import polars as pl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///data/production/database.db")

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

DTYPES = {
    "code": pl.Utf8,
    "url": pl.Utf8,
    "last_modified_t": pl.Int64,
    "product_name": pl.Utf8,
    "packaging_tags": pl.Utf8,
    "categories_tags": pl.Utf8,
    "ingredients_tags": pl.Utf8,
    "ingredients_analysis_tags": pl.Utf8,
    "allergens": pl.Utf8,
    "traces_tags": pl.Utf8,
    "additives_tags": pl.Utf8,
    "nutriscore_grade": pl.Utf8,
    "food_groups_tags": pl.Utf8,
    "states_tags": pl.Utf8,
    "ecoscore_grade": pl.Utf8,
    "nutrient_levels_tags": pl.Utf8,
    "popularity_tags": pl.Utf8,
    "main_category": pl.Utf8,
    "image_url": pl.Utf8,
    "image_small_url": pl.Utf8,
    "energy_100g": pl.Float32,
    "fat_100g": pl.Float32,
    "saturated-fat_100g": pl.Float32,
    "cholesterol_100g": pl.Float32,
    "sugars_100g": pl.Float32,
    "proteins_100g": pl.Float32,
    "salt_100g": pl.Float32,
    "fruits-vegetables-nuts-estimate-from-ingredients_100g": pl.Float32,
    "last_modified_year": pl.Int32,
    "preprocessed_nutriscore_grade": pl.Utf8,
    "preprocessed_ecoscore_grade": pl.Utf8,
    "preprocessed_product_name": pl.Utf8,
    "preprocessed_packaging_tags": pl.Utf8,
    "preprocessed_packaging_tags_lemmatized": pl.Utf8,
    "preprocessed_categories_tags": pl.Utf8,
    "preprocessed_categories_tags_lemmatized": pl.Utf8,
    "preprocessed_ingredients_tags": pl.Utf8,
    "preprocessed_ingredients_tags_lemmatized": pl.Utf8,
    "preprocessed_ingredients_analysis_tags": pl.Utf8,
    "preprocessed_ingredients_analysis_tags_lemmatized": pl.Utf8,
    "preprocessed_nutrient_levels_tags": pl.Utf8,
    "preprocessed_nutrient_levels_tags_lemmatized": pl.Utf8,
    "preprocessed_main_category": pl.Utf8,
    "preprocessed_main_category_lemmatized": pl.Utf8,
    "preprocessed_popularity_tags": pl.Utf8,
    "preprocessed_popularity_tags_lemmatized": pl.Utf8,
    "cluster_text": pl.Utf8,
}


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
    # df = pd.read_csv("./data/production/database.csv")
    df = pl.read_csv("./data/production/database.csv", schema_overrides=DTYPES)

    # 1. Identify the cluster of the reference product
    product_cluster = (
        df.filter(df["code"] == code).select("cluster_text").to_numpy()[0][0]
    )
    target_features = (
        df.filter(df["code"] == code).select(NUMERIC_COLUMNS).to_numpy()[0]
    )

    # 2. Load and merge categorical features
    encoded_categorical_features = pl.read_csv(
        "./data/production/categorical_features.csv"
    )
    cluster_features_combined = df.hstack(encoded_categorical_features)

    # 3. Filter products within the same cluster
    similar_cluster_products = cluster_features_combined.filter(
        pl.col("cluster_text") == product_cluster
    )

    # 4. Filter out products containing the specified allergen, if necessary
    if allergen:
        similar_cluster_products = similar_cluster_products.filter(
            ~pl.col("allergens").str.to_lowercase().str.contains(allergen.lower())
            & ~pl.col("traces_tags").str.to_lowercase().str.contains(allergen.lower())
        )

    # 5. Select only numeric columns for similarity calculation
    similar_cluster_numeric_features = similar_cluster_products.select(
        NUMERIC_COLUMNS
    ).to_numpy()

    # 6. Compute cosine similarity
    similarities = compute_cosine_similarity(
        target_features, similar_cluster_numeric_features
    )

    # 7. Add similarity scores to the filtered products
    similar_cluster_products = similar_cluster_products.with_columns(
        pl.Series("similarity_text", similarities)
    )

    # 8. Sort and return the most similar products
    response = (
        similar_cluster_products.sort(by="similarity_text", descending=True)
        .select(
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
        )
        .head(top_n)
    )

    return response.to_pandas().T.to_json()


async def lazy_find_similar_products_text(code, allergen=None, top_n=10):
    """
    Finds similar products within the same cluster, avoiding those containing a specific allergen.
    Use lazy loading to load CSV files.

    Parameters:
        code (str): Code of the reference product.
        allergen (str): Allergen to avoid, if specified.
        top_n (int): Number of similar products to return.

    Returns:
        DataFrame: Similar products sorted by cosine similarity.
    """
    # Load the dataset lazily
    df = pl.scan_csv("./data/production/database.csv", schema_overrides=DTYPES)

    # 1. Identify the cluster of the reference product
    product_cluster = (
        df.filter(pl.col("code") == str(code))
        .select("cluster_text")
        .collect()
        .to_numpy()[0][0]
    )
    target_features = (
        df.filter(pl.col("code") == str(code))
        .select(NUMERIC_COLUMNS)
        .collect()
        .to_numpy()[0]
    )

    # 2. Load and merge categorical features lazily
    encoded_categorical_features = pl.scan_csv(
        "./data/production/categorical_features.csv"
    )

    # Concatenate the DataFrames horizontally
    cluster_features_combined = pl.concat(
        [df, encoded_categorical_features], how="horizontal"
    )

    # 3. Filter products within the same cluster
    similar_cluster_products = cluster_features_combined.filter(
        pl.col("cluster_text") == product_cluster
    )

    # 4. Filter out products containing the specified allergen, if necessary
    if allergen:
        similar_cluster_products = similar_cluster_products.filter(
            ~pl.col("allergens").str.to_lowercase().str.contains(allergen.lower())
            & ~pl.col("traces_tags").str.to_lowercase().str.contains(allergen.lower())
        )

    # 5. Select only numeric columns for similarity calculation
    similar_cluster_numeric_features = (
        similar_cluster_products.select(NUMERIC_COLUMNS).collect().to_numpy()
    )

    # 6. Compute cosine similarity
    similarities = cosine_similarity(
        [target_features], similar_cluster_numeric_features
    ).flatten()

    # 7. Add similarity scores to the filtered products
    similar_cluster_products = similar_cluster_products.with_columns(
        pl.Series("similarity_text", similarities)
    )

    # 8. Sort and return the most similar products
    response = (
        similar_cluster_products.sort(by="similarity_text", descending=True)
        .select(
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
        )
        .head(top_n)
        .collect()
    )

    return response.to_pandas().T.to_json()


async def sql_find_similar_products_text(code, allergen=None, top_n=10):
    """
    Finds similar products within the same cluster, avoiding those containing a specific allergen.
    Uses SQLite database instead of CSV files.

    Parameters:
        code (str): Code of the reference product.
        allergen (str): Allergen to avoid, if specified.
        top_n (int): Number of similar products to return.

    Returns:
        DataFrame: Similar products sorted by cosine similarity.
    """
    # Charger les deux DataFrames
    # df_polars = pl.read_csv("./data/production/database.csv", schema_overrides=DTYPES)
    # df_polars2 = pl.read_csv("./data/production/categorical_features.csv")

    # Ajouter la colonne 'code' de product_data dans categorical_features
    # df_polars2 = df_polars2.with_columns(df_polars["code"].alias("code"))

    # Sauvegarder dans la base de données
    DATABASE_URL = "sqlite:///db/foodstuffs-recommendation.db"
    engine = create_engine(DATABASE_URL)

    # df_polars.to_pandas().to_sql(
    #     "product_data", con=engine, if_exists="replace", index=False
    # )
    # df_polars2.to_pandas().to_sql(
    #     "categorical_features", con=engine, if_exists="replace", index=False
    # )
    # 1. Identifier le cluster du produit de référence
    with engine.connect() as connection:
        query = text("SELECT cluster_text FROM product_data WHERE code = :code")
        result = connection.execute(query, {"code": str(code)}).fetchone()
        product_cluster = result[0]

        # Obtenir les caractéristiques numériques du produit cible
        quoted_columns = [f'"{col}"' for col in NUMERIC_COLUMNS]
        query = text(
            "SELECT "
            + ", ".join(quoted_columns)
            + " FROM product_data WHERE code = :code"
        )
        target_features = connection.execute(query, {"code": str(code)}).fetchone()

    # 2. Charger les caractéristiques catégorielles (si vous les avez stockées dans une table séparée)
    with engine.connect() as connection:
        query = text("SELECT * FROM categorical_features")
        encoded_categorical_features = pd.read_sql(query, connection)

    # Convertir en Polars
    encoded_categorical_features = pl.from_pandas(encoded_categorical_features)

    # 3. Charger et fusionner les caractéristiques du cluster avec les caractéristiques catégorielles
    with engine.connect() as connection:
        query = text("SELECT * FROM product_data WHERE cluster_text = :cluster_text")
        similar_cluster_products_df = pd.read_sql(
            query, connection, params={"cluster_text": product_cluster}
        )

    # Convertir en Polars
    similar_cluster_products = pl.from_pandas(similar_cluster_products_df)

    # 4. Filtrer les produits contenant l'allergène, si nécessaire
    if allergen:
        similar_cluster_products = similar_cluster_products.filter(
            ~pl.col("allergens").str.to_lowercase().str.contains(allergen.lower())
            & ~pl.col("traces_tags").str.to_lowercase().str.contains(allergen.lower())
        )

    # 5. Sélectionner uniquement les colonnes numériques pour le calcul de la similarité
    similar_cluster_numeric_features = similar_cluster_products.select(
        NUMERIC_COLUMNS
    ).to_numpy()

    # 6. Calculer la similarité cosinus
    similarities = cosine_similarity(
        [target_features], similar_cluster_numeric_features
    ).flatten()

    # 7. Ajouter les scores de similarité aux produits filtrés
    similar_cluster_products = similar_cluster_products.with_columns(
        pl.Series("similarity_text", similarities)
    )

    # 8. Trier et retourner les produits les plus similaires
    response = (
        similar_cluster_products.sort(by="similarity_text", descending=True)
        .select(
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
        )
        .head(top_n)
    )

    return response.to_pandas().T.to_json()


async def sql_without_cat_feat_find_similar_products_text(
    code, allergen=None, top_n=10
):
    """
    Finds similar products within the same cluster, avoiding those containing a specific allergen.
    Use SQL database.

    Parameters:
        code (str): Code of the reference product.
        allergen (str): Allergen to avoid, if specified.
        top_n (int): Number of similar products to return.

    Returns:
        DataFrame: Similar products sorted by cosine similarity.
    """
    df_polars = pl.read_csv("./data/production/database.csv", schema_overrides=DTYPES)

    # 2. Convertir le DataFrame Polars en DataFrame Pandas
    df_pandas = df_polars.to_pandas()

    # 3. Créer une connexion SQLAlchemy à votre base de données
    # Remplacez la chaîne de connexion par celle correspondant à votre base de données
    DATABASE_URL = "sqlite:///foodstuffs-recommendation.db"
    engine = create_engine(DATABASE_URL)

    # 4. Enregistrer le DataFrame Pandas dans la base de données
    df_pandas.to_sql("product_data", con=engine, if_exists="replace", index=False)

    # 1. Identify the cluster of the reference product
    with engine.connect() as connection:
        result = connection.execute(
            text(
                """
            SELECT cluster_text, {}
            FROM database
            WHERE code = :code
        """.format(", ".join(NUMERIC_COLUMNS))
            ),
            {"code": code},
        )
        row = result.fetchone()
        product_cluster = row["cluster_text"]
        target_features = np.array([row[col] for col in NUMERIC_COLUMNS])

    # 2. Load and merge categorical features
    with engine.connect() as connection:
        result = connection.execute(
            text("""
            SELECT *
            FROM categorical_features
        """)
        )
        encoded_categorical_features = pd.DataFrame(
            result.fetchall(), columns=result.keys()
        )

    # 3. Filter products within the same cluster
    with engine.connect() as connection:
        query = """
            SELECT *
            FROM database
            WHERE cluster_text = :product_cluster
        """
        if allergen:
            query += """
                AND LOWER(allergens) NOT LIKE :allergen
                AND LOWER(traces_tags) NOT LIKE :allergen
            """
        result = connection.execute(
            text(query),
            {"product_cluster": product_cluster, "allergen": f"%{allergen.lower()}%"},
        )
        similar_cluster_products = pd.DataFrame(
            result.fetchall(), columns=result.keys()
        )

    # 4. Select only numeric columns for similarity calculation
    similar_cluster_numeric_features = similar_cluster_products[NUMERIC_COLUMNS].values

    # 5. Compute cosine similarity
    similarities = cosine_similarity(
        [target_features], similar_cluster_numeric_features
    ).flatten()

    # 6. Add similarity scores to the filtered products
    similar_cluster_products["similarity_text"] = similarities

    # 7. Sort and return the most similar products
    response = similar_cluster_products.sort_values(
        by="similarity_text", ascending=False
    ).head(top_n)

    return response.T.to_json()


def find_similar_products_img(
    product_code,
    top_n,
):
    """
    Finds similar products in the same cluster based on cosine similarity, using the product code.
    The returned DataFrame contains only the products from the same cluster and a 'similarity_img' column
    with similarity scores, sorted in descending order.

    Parameters:
        product_code (str/int): The code of the product for which to find similar products.
        top_n (int): Number of similar products to return.

    Returns:
        str: JSON containing the similar products.
    """
    product_code_column = "code"
    embedding_prefix = "embedding_"
    cluster_column = "cluster_emb"

    # Load the DataFrame
    df = pd.read_csv("./data/clean_dataset_clusters.csv")

    print(df[[product_code_column]].dtypes)

    # Check that the required columns exist
    embedding_columns = [col for col in df.columns if col.startswith(embedding_prefix)]
    if not embedding_columns:
        raise KeyError(
            f"Columns with the prefix '{embedding_prefix}' must exist in the DataFrame."
        )

    # Find the entry for the given product
    target_row = df[df[product_code_column] == product_code]
    if target_row.empty:
        raise ValueError(
            f"The product with the code '{product_code}' does not exist in the DataFrame."
        )

    # Get the embedding and cluster of the target product
    target_embedding = target_row[embedding_columns].values
    if target_embedding.shape[0] == 0 or np.isnan(target_embedding).any():
        raise ValueError(
            f"The embedding for the product with the code '{product_code}' is empty or invalid."
        )
    target_cluster = target_row.iloc[0][cluster_column]

    # Filter products belonging to the same cluster
    cluster_products = df[df[cluster_column] == target_cluster]

    # Remove the target product row from the cluster to avoid self-comparison
    cluster_products = cluster_products[
        cluster_products[product_code_column] != product_code
    ]

    # Compute cosine similarities with products in the cluster
    cluster_embeddings = cluster_products[embedding_columns].values
    similarities = cosine_similarity(target_embedding, cluster_embeddings).flatten()

    # Add the 'similarity_img' column to the DataFrame of products in the same cluster
    cluster_products = cluster_products.copy()
    cluster_products["similarity_img"] = similarities

    # Sort by similarity and keep the top_n products
    cluster_products = cluster_products.sort_values(
        by="similarity_img", ascending=False
    )
    response = cluster_products.drop(columns=embedding_columns).head(top_n)

    return response.T.to_json()
