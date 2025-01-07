import sys
import os

# todo Uptimise
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.utils.config import load_config
from training.utils.logger import setup_logger

import os
import pandas as pd
from tflite_support.task import vision  ## Run on python 3.8
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


# Initialize the embedder model for images
image_embedder = vision.ImageEmbedder.create_from_file(path)


def get_image_embedding(image_path):
    """
    Generates an embedding for a local image using ImageEmbedder.

    Parameters:
        image_path (str): The local path to the image.

    Returns:
        np.ndarray: The feature vector of the image.
    """
    try:
        image = Image.open(image_path)
        image_array = np.array(image)

        # Extract the embedding
        tensor_image = vision.TensorImage.create_from_array(image_array)
        embedding_result = image_embedder.embed(tensor_image)
        feature_vector = embedding_result.embeddings[0].feature_vector
        return feature_vector
    except Exception as e:
        # print(f"Error processing image {image_path}: {e}")
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

    # Check if embedding is already loaded
    embedding_columns = [col for col in df.columns if col.startswith("embedding_")]
    if embedding_columns:
        logger.info("Embeddings have been already loaded.")
        return df

    for index, row in df.iterrows():
        code = row[code_column]
        image_path = os.path.join(save_dir, f"{code}.jpg")

        if os.path.exists(image_path):
            # Generate embedding if the image exists
            embedding = get_image_embedding(image_path)
            if embedding:
                embedding_array = embedding.value
            else:
                embedding_array = None
        else:
            embedding = None
            embedding_array = None

        if index == img_download:
            print(f"{img_download} images processed out of {len(df)}")
            img_download += 25000

        embeddings.append(embedding)
        embeddings_array.append(embedding_array)

    # Add embeddings to the DataFrame
    # df["image_embedding"] = embeddings
    df["embedding_array"] = embeddings_array

    # Check that the `embedding_array` column contains valid values
    valid_embeddings = df["embedding_array"].apply(
        lambda x: isinstance(x, np.ndarray) and len(x) > 0
    )

    # Get the length of valid embeddings
    if valid_embeddings.any():
        len_valid_embeddings = len(df.loc[valid_embeddings, "embedding_array"].iloc[0])
    else:
        raise ValueError("No valid embedding found in the 'embedding_array' column.")

    # Create a new DataFrame for embedding columns
    embedding_columns = pd.DataFrame(
        df["embedding_array"]
        .apply(
            lambda x: x
            if isinstance(x, np.ndarray) and len(x) == len_valid_embeddings
            else [None] * len_valid_embeddings
        )
        .tolist(),
        columns=[f"embedding_{i}" for i in range(len_valid_embeddings)],
    )

    # Concatenate the new columns with the original DataFrame
    df = pd.concat([df.drop(columns=["embedding_array"]), embedding_columns], axis=1)

    df.to_csv(df_path, index=False)


def main(config_path: str):
    # Load configuration from file
    config = load_config(config_path)

    df_path = config["data"]["production_data_path"]
    code_column = config["data"]["code_column"]
    save_dir = config["data"]["images_path"]

    get_all_image_embeddings(
        df_path,
        code_column=code_column,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    logger.info("Embedding start")
    import argparse

    parser = argparse.ArgumentParser(description="download images")
    parser.add_argument("--config", required=True, help="Configuration file path")
    args = parser.parse_args()

    main(args.config)
