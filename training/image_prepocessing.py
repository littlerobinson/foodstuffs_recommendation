from utils.config import load_config
from utils.logger import setup_logger

import os
import pandas as pd
from tflite_support.task import vision ## Run on python 3.8
import io
from PIL import Image
import numpy as np

import kagglehub

logger = setup_logger()

# Download latest version
path = kagglehub.model_download(
    "google/mobilenet-v3/tfLite/small-100-224-feature-vector-metadata"
)
path = path + "/1.tflite"
print("Path to model files:", path)


# Initialise le modèle d'embedder pour les images
image_embedder = vision.ImageEmbedder.create_from_file(path)

def get_image_embedding(image_path):
    """
    Génère un embedding pour une image locale en utilisant ImageEmbedder.

    Parameters:
        image_path (str): Le chemin local de l'image.

    Returns:
        np.ndarray: Le vecteur de caractéristiques de l'image.
    """
    try:
        image = Image.open(image_path)
        image_array = np.array(image)

        # Extraction de l'embedding
        tensor_image = vision.TensorImage.create_from_array(image_array)
        embedding_result = image_embedder.embed(tensor_image)
        feature_vector = embedding_result.embeddings[0].feature_vector
        return feature_vector
    except Exception as e:
        # print(f"Erreur lors du traitement de l'image {image_path}: {e}")
        return None


def get_all_image_embeddings(df_path, code_column, save_dir):
    """
    Génère des embeddings pour toutes les images déjà téléchargées localement.

    Parameters:
        df (DataFrame): Le DataFrame contenant les codes uniques pour retrouver les images.
        code_column (str): Le nom de la colonne contenant les codes uniques.
        save_dir (str): Le dossier où sont enregistrées les images.

    Returns:
        DataFrame: Un DataFrame contenant les embeddings de chaque image dans une colonne `image_embedding`.
    """
    df = pd.read_csv(df_path)
    embeddings = []
    embeddings_array = []
    img_download = 25000

    for index, row in df.iterrows():
        code = row[code_column]
        image_path = os.path.join(save_dir, f"{code}.jpg")

        if os.path.exists(image_path):
            # Génère l'embedding si l'image existe
            embedding = get_image_embedding(image_path)
            if embedding:
                embedding_array = embedding.value
            else:
                embedding_array = None
        else:
            # print(f"L'image avec le code {code} est manquante dans {save_dir}.")
            embedding = None
            embedding_array = None

        if index == img_download:
            print(f"{img_download} images traitées sur {len(df)}")
            img_download += 25000

        embeddings.append(embedding)
        embeddings_array.append(embedding_array)

    # Ajoute les embeddings au DataFrame
    # df["image_embedding"] = embeddings
    df["embedding_array"] = embeddings_array

    # Vérifier que la colonne embedding_array contient des valeurs valides
    valid_embeddings = df["embedding_array"].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)

    # Obtenir la longueur des embeddings valides
    if valid_embeddings.any():
        len_valid_embeddings = len(df.loc[valid_embeddings, "embedding_array"].iloc[0])
        print(len_valid_embeddings)
    else:
        raise ValueError("Aucun embedding valide trouvé dans la colonne 'embedding_array'.")

    # Create a new DataFrame for embedding columns
    embedding_columns = pd.DataFrame(
        df["embedding_array"].apply(
            lambda x: x if isinstance(x, np.ndarray) and len(x) == len_valid_embeddings else [None] * len_valid_embeddings
        ).tolist(),
        columns=[f"embedding_{i}" for i in range(len_valid_embeddings)]
    )

    # Concatenate the new columns with the original DataFrame
    df = pd.concat([df.drop(columns=["embedding_array"]), embedding_columns], axis=1)

    df.to_csv("data/clean_dataset_with_embeddings.csv", index=False)

def main(config_path: str):
    # Load configuration from file
    config = load_config(config_path)

    df_path = config["data"]["clean_data_path"]
    code_column = config["data"]["code_column"]
    save_dir = config["data"]["images_path"]

    get_all_image_embeddings(df_path, code_column=code_column, save_dir=save_dir,)


if __name__ == "__main__":
    logger.info("Embedding start")
    import argparse

    parser = argparse.ArgumentParser(description="Téléchargement d'images")
    parser.add_argument(
        "--config", required=True, help="Chemin du fichier de configuration"
    )
    args = parser.parse_args()

    main(args.config)
