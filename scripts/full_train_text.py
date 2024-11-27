import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import subprocess
import argparse


def update_config(config_path, n_clusters, encoding_method_name):
    """
    Update the configuration file with the given parameters.

    Parameters:
    config_path (str): The path to the configuration file.
    n_clusters (int): The number of clusters.
    encoding_method_name (str): The encoding method name.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["training"]["n_clusters"] = n_clusters
    config["training"]["encoding_method_name"] = encoding_method_name

    with open(config_path, "w") as file:
        yaml.safe_dump(config, file)


def main(config_path):
    # Define the range of n_clusters and encoding methods
    n_clusters_range = range(100, 1000, 100)
    encoding_methods = ["svd", "pca"]

    # Iterate over the combinations of n_clusters and encoding methods
    for n_clusters in n_clusters_range:
        for encoding_method in encoding_methods:
            # Update the configuration file
            update_config(config_path, n_clusters, encoding_method)

            # Run the main script with the updated configuration
            subprocess.run(
                ["python", "training/main.py", "--config", config_path, "--mlflow"]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the main script with different parameters."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the configuration file."
    )
    args = parser.parse_args()

    main(args.config)
