# Variables
ML_CONFIG_PATH = training/config.yaml
SECRET_FILE = secrets.sh

# Build and run docker env
docker_up:
	@echo "Build and run docker env..."
	docker compose up --build

# Stop containers
docker_down:
	@echo "Stop containers..."
	docker compose down

# Visualize docker logs
docker_logs:
	docker compose logs -f

# Export secrets
export_secrets:
	@echo "Export secrets..."
	@chmod +x $(SECRET_FILE)
	@bash -c "source $(SECRET_FILE)"

# Run docker-cli
run_docker_cli:
	@echo "Run docker-cli..."
	docker compose exec python-cli bash

# From here the code the following command has to be run in the docker previouly created

# Install dependencies with Poetry
install_deps:
	@echo "Installing dependencies with Poetry..."
	poetry install
	poetry run python -m spacy download en_core_web_sm

# Load data from raw database
load_data:
	@echo "Load raw data..."
	poetry run python training/main.py --load_data --config $(ML_CONFIG_PATH)

# Run MLFlow
mlflow: export_secrets
	@echo "Run MLFlow..."
	poetry run python training/main.py --mlflow --config $(ML_CONFIG_PATH)

# Run the main machine learning pipeline
run: export_secrets
	@echo "Running the main project pipeline..."
	# source ./secrets.sh
	$(MAKE) load_data
	$(MAKE) mlflow

# Download all the images
download_images:
	@echo "Downloading all the images..."
	poetry run python scripts/images_downloader.py --config $(ML_CONFIG_PATH)

# Open another terminal
# Run python cli 3.8
run_add_embeddings:
	@echo "Add embeddings on the dataset from the image files using python 3.8 ..."
	docker compose exec python-cli-3.8 python scripts/image_prepocessing.py --config $(ML_CONFIG_PATH)

# # Add embeddings: Has to be run on python 3.8
# run_add_embeddings:
# 	@echo "Add embeddings on the dataset from the image files..."
# 	poetry run python scripts/image_prepocessing.py --config $(ML_CONFIG_PATH)

# Come bach on the terminal with python cli 3.12 


# Run clustering on image
run_image_clustering:
	@echo "Running clustering on image..."
	poetry run python scripts/image_clustering.py --config $(ML_CONFIG_PATH)

# Clean temporary and cache files
clean:
	@echo "Cleaning up temporary files..."
	rm -rf __pycache__ */__pycache__
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

# Run all tests in the project with Poetry
run_tests:
	@echo "Running tests..."
	poetry run python -m unittest discover -s tests/api

	poetry run python -m unittest discover -s tests/training

# Display available Makefile commands
help:
	@echo "Available Makefile commands:"
	@echo "  make install_deps    - Install dependencies with Poetry"
	@echo "  make run             - Run the main machine learning pipeline"
	@echo "  make clean           - Remove temporary and cache files"
	@echo "  make run_tests       - Run unit and integration tests"
