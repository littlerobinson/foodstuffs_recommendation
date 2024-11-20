from utils.config import load_config
from handlers.download_image import download_all_images
from utils.logger import setup_logger

logger = setup_logger()

def main(config_path: str):
    # Load configuration from file
    config = load_config(config_path)

    df_path = config["data"]["clean_data_path"]
    image_url_column = config["data"]["image_url_column"]
    code_column = config["data"]["code_column"]
    save_dir = config["data"]["images_path"]

    logger.info(f"Configuration chargée depuis : {config_path}")
    logger.info(f"Données lues depuis : {df_path}")
    logger.info(f"Colonne des URLs d'images : {image_url_column}")
    logger.info(f"Colonne des codes : {code_column}")
    logger.info(f"Dossier de sauvegarde : {save_dir}")

    # Download all images from the DataFrame
    download_all_images(
        df_path,
        image_url_column=image_url_column,
        code_column=code_column,
        save_dir=save_dir,
    )

if __name__ == "__main__":
    logger.info("🚀  Start downloading 🚀")
    import argparse

    parser = argparse.ArgumentParser(description="Téléchargement d'images")
    parser.add_argument(
        "--config", required=True, help="Chemin du fichier de configuration"
    )
    args = parser.parse_args()

    main(args.config)
