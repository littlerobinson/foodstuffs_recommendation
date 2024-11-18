import argparse
import sys
from pipelines.data_pipeline import build_pipeline
from utils.config import load_config
from utils.logger import setup_logger

logger = setup_logger()


def exit_program():
    print("Exiting the program...")
    sys.exit(0)


def main(config_path: str):
    # load config variables
    config = load_config(config_path)

    rawdata = build_pipeline(
        config["data"]["raw_data_path"], config["data"]["processed_data_path"]
    )
    if rawdata.empty:
        logger.warning("Raw data is empty. Check the file path or filters.")
        exit_program()
    else:
        logger.info(f"Raw data loaded successfully: {len(rawdata)} rows available.")


if __name__ == "__main__":
    logger.info("🚀  Foodstuffs Recommendation Launched. 🚀")

    parser = argparse.ArgumentParser(description="Starting the ML pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Set the config file path",
    )
    args = parser.parse_args()
    main(args.config)
