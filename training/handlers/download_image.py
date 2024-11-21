import pandas as pd
import requests
import os


def download_image_if_not_exists(image_url, code, save_dir):
    """
    Télécharge une image depuis une URL si elle n'existe pas déjà dans le dossier local.

    Parameters:
        image_url (str): L'URL de l'image.
        code (str): Le code unique pour nommer le fichier.
        save_dir (str): Le dossier où enregistrer les images.

    Returns:
        str: Le chemin du fichier local de l'image, ou None si le téléchargement a échoué.
    """
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"{code}.jpg")

    if not os.path.exists(image_path):  # Vérifie si l'image existe déjà
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
                # print(f"Image téléchargée et enregistrée: {image_path}")
            else:
                # print(f"Échec du téléchargement pour {image_url}. Code HTTP: {response.status_code}")
                return None
        except Exception as e:
            # print(f"Erreur lors du téléchargement de {image_url}: {e}")
            return None
    else:
        # print(f"L'image existe déjà: {image_path}")
        next

    return image_path


def download_all_images(df_path, image_url_column, code_column, save_dir):
    """
    Télécharge toutes les images à partir des URLs fournies dans un DataFrame.

    Parameters:
        df_path (str): Le chemin du DataFrame contenant les URLs des images et les codes uniques.
        image_url_column (str): Le nom de la colonne contenant les URLs des images.
        code_column (str): Le nom de la colonne contenant les codes uniques.
        save_dir (str): Le dossier où enregistrer les images.

    Returns:
        None
    """
    df = pd.read_csv(df_path)
    img_download = 25_000
    for index, row in df.iterrows():
        image_url = row[image_url_column]
        code = row[code_column]
        if index == img_download:
            print(f"{img_download} images traitées sur {len(df)}")
            img_download += 25000

        # Télécharge l'image
        download_image_if_not_exists(image_url, code, save_dir)
