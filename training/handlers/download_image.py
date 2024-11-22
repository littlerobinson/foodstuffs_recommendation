import pandas as pd
import requests
import os


def download_image_if_not_exists(image_url, code, save_dir):
    """
    Downloads an image from a URL if it does not already exist in the local directory.

    Parameters:
        image_url (str): The URL of the image.
        code (str): The unique code to name the file.
        save_dir (str): The directory where the images will be saved.

    Returns:
        str: The local file path of the image, or None if the download failed.
    """
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"{code}.jpg")

    if not os.path.exists(image_path):  # Checks if the image already exists
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
            else:
                return None
        except Exception as e:
            return None
    else:
        next

    return image_path


def download_all_images(df_path, image_url_column, code_column, save_dir):
    """
    Downloads all images from the URLs provided in a DataFrame.

    Parameters:
        df_path (str): The path to the DataFrame containing the image URLs and unique codes.
        image_url_column (str): The name of the column containing the image URLs.
        code_column (str): The name of the column containing the unique codes.
        save_dir (str): The directory where the images will be saved.

    Returns:
        None
    """
    df = pd.read_csv(df_path)
    img_download = 25_000
    for index, row in df.iterrows():
        image_url = row[image_url_column]
        code = row[code_column]
        if index == img_download:
            print(f"{img_download} images processed out of {len(df)}")
            img_download += 25000

        # Download the image
        download_image_if_not_exists(image_url, code, save_dir)
