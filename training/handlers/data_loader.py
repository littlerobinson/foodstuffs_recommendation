import pandas as pd
from utils.logger import setup_logger

logger = setup_logger()


def extract_raw_data(csv_file_path, chunk_size=10000, countries_tags="en:france"):
    """
    Extracts raw data from the compressed OpenFoodFacts file and filters products for a given country.

    Parameters:
    - file_path (str): Path to the CSV or compressed CSV data file.
    - chunk_size (int): Size of each chunk to read to avoid memory overload.
    - countries_tags (str): Select Country tags to get products from specific country.

    Returns:
    - rawdata (DataFrame): Dataset filtered for products with ingredients.
    """
    logger.info("Extracting raw data 🏗️.")
    try:
        filtered_chunks_list = []

        # Chargement par chunks pour gérer les gros fichiers
        for chunk in pd.read_csv(
            csv_file_path,
            chunksize=chunk_size,
            compression="gzip",
            sep="\t",
            engine="python",
            quoting=3,
        ):
            filtered_chunks = chunk[
                (chunk["countries_tags"].str.contains(countries_tags, na=False))
                & (chunk["ingredients_tags"].notna())
            ]
            filtered_chunks_list.append(filtered_chunks)

        if filtered_chunks_list:
            rawdata = pd.concat(filtered_chunks_list, axis=0)
        else:
            rawdata = pd.DataFrame()  # return empty dataframme if is empty

        logger.info(
            f"Data loading and filtering completed: {len(rawdata)} rows selected."
        )
        return rawdata

    except Exception as e:
        logger.error(f"Error loading data : {e}")
        return pd.DataFrame()


def load_dataset(dataset_path, nrows=None):
    if nrows is None:
        return pd.read_csv(dataset_path)
    else:
        return pd.read_csv(dataset_path, nrows=nrows)
