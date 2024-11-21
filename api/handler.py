import os
import sys

# Ajouter le répertoire parent au sys.path
sys.path.append(os.path.abspath(".."))

import io
import re
from collections import Counter

import kagglehub
import numpy as np

# import en_core_web_sm
import pandas as pd
import plotly.express as px

# Image embedding
import requests
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tflite_support.task import vision

# from spacy.lang.en.stop_words import STOP_WORDS
from training.classes.language_tools import TextProcessing

# Download latest version
path = kagglehub.model_download(
    "google/mobilenet-v3/tfLite/small-100-224-feature-vector-metadata"
)
path = path + "/1.tflite"
print("Path to model files:", path)

# Initialise le modèle d'embedder pour les images
image_embedder = vision.ImageEmbedder.create_from_file(path)


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
    "product_name",
    "packaging_tags",
    "categories_tags",
    "ingredients_tags",
    "ingredients_analysis_tags",
    "main_category",
]


def compute_cosine_similarity(target_embedding, all_embeddings):
    """
    Calcule la similarité cosinus entre une image cible et toutes les autres images.

    Parameters:
        target_embedding (np.ndarray): Embedding de l'image cible.
        all_embeddings (np.ndarray): Tableau des embeddings d'autres images.

    Returns:
        np.ndarray: Scores de similarité cosinus avec l'image cible.
    """
    similarities = cosine_similarity(target_embedding.reshape(1, -1), all_embeddings)
    return similarities.flatten()


def find_similar_products_text(
    code, df, allergen=None, top_n=10, encoding_method=encode_categorical_data_with_svd
):
    """
    Recherche des produits similaires dans le même cluster en évitant ceux contenant un allergène spécifique.

    Parameters:
        df (DataFrame): Le DataFrame contenant les données produits.
        code (str): Code du produit de référence.
        allergen (str): Allergène à éviter, si spécifié.
        top_n (int): Nombre de produits similaires à retourner.
        encoding_method (function): La méthode d'encodage des données catégorielles.

    Returns:
        DataFrame: Les produits similaires triés par similarité de cosinus.
    """
    # 1. Identifier le cluster du produit de référence
    product_cluster = df.loc[df["code"] == code, "cluster"].values[0]

    # 2. Filtrer les produits du même cluster
    similar_cluster_products = df[df["cluster"] == product_cluster]

    # 3. Si un allergène est spécifié, exclure les produits qui le contiennent
    if allergen:
        similar_cluster_products = similar_cluster_products[
            ~similar_cluster_products["allergens"]
            .fillna("")
            .str.contains(allergen, case=False)
            & ~similar_cluster_products["traces_tags"]
            .fillna("")
            .str.contains(allergen, case=False)
        ]

    # 4. Encoder les colonnes catégorielles
    encoded_categorical_features = encoding_method(
        similar_cluster_products, CATEGORIAL_COLUMNS
    )

    # Ajouter la colonne code à encoded_categorical_features
    encoded_categorical_features["code"] = similar_cluster_products["code"].reset_index(
        drop=True
    )

    # 5. Extraire les caractéristiques du produit cible
    target_product_row = encoded_categorical_features.loc[
        encoded_categorical_features["code"] == code
    ]
    target_product_features = target_product_row.drop(columns=["code"]).values
    target_numeric_features = similar_cluster_products.loc[
        similar_cluster_products["code"] == code, NUMERIC_COLUMNS
    ].values

    # 6. Supprimer la ligne du produit cible dans encoded_categorical_features et dans les colonnes numériques
    cluster_features = encoded_categorical_features[
        encoded_categorical_features["code"] != code
    ].drop(columns=["code"])
    cluster_numeric_features = similar_cluster_products[NUMERIC_COLUMNS].drop(
        index=similar_cluster_products[similar_cluster_products["code"] == code].index
    )

    # 7. Combiner les caractéristiques numériques et catégorielles
    target_features = np.hstack([target_numeric_features, target_product_features])
    cluster_features_combined = np.hstack(
        [cluster_numeric_features.values, cluster_features.values]
    )

    # 8. Calcul des similarités de cosinus
    similarities = cosine_similarity(
        target_features.reshape(1, -1), cluster_features_combined
    ).flatten()

    # 9. Réinitialiser l'index pour aligner la longueur des données
    similar_cluster_products = similar_cluster_products[
        similar_cluster_products["code"] != code
    ].reset_index(drop=True)

    # 10. Ajouter la similarité aux produits similaires
    similar_cluster_products = similar_cluster_products.assign(
        similarity_text=similarities
    )

    # 11. Trier et retourner les produits les plus similaires
    return similar_cluster_products.sort_values(by="similarity_text", ascending=False)[
        ["code", "url", "product_name", "cluster", "similarity_text"]
    ].head(top_n)


def find_similar_products_img(
    product_code,
    df,
    product_code_column="code",
    embedding_column="embedding_array",
    cluster_column="cluster_emb",
    top_n=10,
):
    """
    TODO: to refactorize
    Trouve les produits similaires dans le même cluster basé sur la similarité cosinus, en utilisant le code du produit.
    Le DataFrame retourné contient uniquement les produits du même cluster et une colonne 'similarity'
    avec les scores de similarité, triés par ordre décroissant.

    Parameters:
        product_code (str/int): Le code du produit pour lequel trouver des produits similaires.
        df (pd.DataFrame): DataFrame contenant les informations sur les produits, leurs clusters et embeddings.
        product_code_column (str): Nom de la colonne contenant les codes des produits.
        embedding_column (str): Nom de la colonne contenant les embeddings des produits.
        cluster_column (str): Nom de la colonne contenant les clusters des produits.
        top_n (int): Nombre de produits similaires à retourner.

    Returns:
        pd.DataFrame: DataFrame contenant uniquement les produits du même cluster, avec une colonne 'similarity'
                      et trié par ordre décroissant de similarité.
    """
    # Vérifier que les colonnes nécessaires existent
    if not all(
        col in df.columns
        for col in [product_code_column, embedding_column, cluster_column]
    ):
        raise KeyError(
            f"Les colonnes {product_code_column}, {embedding_column}, et {cluster_column} doivent exister dans le DataFrame."
        )

    # Trouver l'entrée pour le produit donné
    target_row = df[df[product_code_column] == product_code]
    if target_row.empty:
        raise ValueError(
            f"Le produit avec le code '{product_code}' n'existe pas dans le DataFrame."
        )
    elif target_row[[embedding_column]].empty:
        raise ValueError(
            f"L'image ou le embedding du produit avec le code '{product_code}' est vide."
        )

    # Obtenir l'embedding et le cluster du produit cible
    target_embedding = target_row.iloc[0][embedding_column]
    target_cluster = target_row.iloc[0][cluster_column]

    # Filtrer les produits appartenant au même cluster
    cluster_products = df[df[cluster_column] == target_cluster]

    # Supprimer la ligne du produit cible du cluster pour éviter la comparaison avec lui-même
    cluster_products = cluster_products.drop(index=target_row.index)

    # Calculer les similarités cosinus avec les produits du cluster
    similarities = []
    for _, row in cluster_products.iterrows():
        embedding = row[embedding_column].reshape(1, -1)
        try:
            similarity = compute_cosine_similarity(target_embedding, embedding)
            similarities.append((row[product_code_column], similarity[0]))
        except ValueError as e:
            # Si l'embedding est vide ou incorrect
            print(
                f"Erreur de similarité pour le produit avec le code {row[product_code_column]} : {e}"
            )
            similarities.append((row[product_code_column], None))

    # Ajouter la colonne 'similarity' au DataFrame des produits du même cluster
    cluster_products["similarity_img"] = [sim[1] for sim in similarities]

    # Filtrer les lignes qui ont des valeurs de similarité non-nulles et trier par similarité
    cluster_products = cluster_products[cluster_products["similarity_img"].notnull()]
    cluster_products = cluster_products.sort_values(
        by="similarity_img", ascending=False
    )
    cluster_products.drop(columns=["image_embedding", "embedding_array"], inplace=True)

    return cluster_products.head(top_n)
