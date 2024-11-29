# Variables
ML_CONFIG_PATH = training/config.yaml
SECRET_FILE = secrets.sh

# Run the main machine learning pipeline (default)
default: help

###############################################################################
# Docker Management
###############################################################################

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

###############################################################################
# Secrets Management
###############################################################################

# Export secrets
export_secrets:
	@echo "Export secrets..."
	@chmod +x $(SECRET_FILE)
	@bash -c "source $(SECRET_FILE)"

###############################################################################
# Command Line Access
###############################################################################

# Run docker-cli
run_docker_cli:
	@echo "Run docker-cli..."
	docker compose exec python-cli bash

###############################################################################
# Dependency Management
###############################################################################

# Install Python dependencies with Poetry
install_deps:
	@echo "Installing dependencies with Poetry..."
	poetry install
	poetry run python -m spacy download en_core_web_sm

###############################################################################
# Data Loading and Processing
###############################################################################

# Load data from raw database
load_data:
	@echo "Load raw data..."
	poetry run python training/main.py --load_data --config $(ML_CONFIG_PATH)

# Download all products images
download_images:
	@echo "Downloading all the images..."
	poetry run python scripts/images_downloader.py --config $(ML_CONFIG_PATH)

# Run python cli 3.8
run_add_embeddings:
	@echo "Add embeddings on the dataset from the image files using python 3.8 ..."
	docker compose exec python-cli-3.8 python scripts/image_prepocessing.py --config $(ML_CONFIG_PATH)

# Run python cli 3.8 with gpu compatibility (WIP)
run_add_embeddings_gpu:
	@echo "Add embeddings on the dataset from the image files using python 3.8 ..."
	docker run --rm -it -v "./:/app" foodstuffs-recommendation-python-gpu-cli-3.8 scripts/image_prepocessing.py --config $(ML_CONFIG_PATH)

###############################################################################
# Machine Learning Pipeline
###############################################################################

# Run Training with MLFlow monitoring
train_mlflow: export_secrets
	@echo "Run MLFlow..."
	poetry run python training/main.py --mlflow --config $(ML_CONFIG_PATH)

train_full_text_mlflow: export_secrets
	@echo "Run Full MLFlow..."
	poetry run python scripts/full_train_text.py --config $(ML_CONFIG_PATH)

# Run the main machine learning training pipeline for text
train: export_secrets
	@echo "Running the main project pipeline..."
	# source ./secrets.sh
	$(MAKE) load_data
	$(MAKE) mlflow

# Run clustering on image
image_clustering:
	@echo "Running clustering on image..."
	poetry run python scripts/image_clustering.py --config $(ML_CONFIG_PATH)

###############################################################################
# Database and Clustering
###############################################################################

# Create text API database
create_text_api_database:
	@echo "Create the API database for text script..."
	poetry run python scripts/create_api_database.py --config $(ML_CONFIG_PATH)

###############################################################################
# Testing and Cleanup
###############################################################################

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

###############################################################################
# Help
###############################################################################

# Display available Makefile commands
# Display available Makefile commands
help:
	@echo "📦 Docker Management:"
	@echo "  make docker_up            - Build and run the Docker environment"
	@echo "  make docker_down          - Stop all running Docker containers"
	@echo "  make docker_logs          - View real-time logs from Docker containers"
	@echo ""
	@echo "🔑 Secrets Management:"
	@echo "  make export_secrets       - Load and export environment secrets from secrets.sh"
	@echo ""
	@echo "🖥️ Command Line Access:"
	@echo "  make run_docker_cli       - Open a shell in the Python CLI Docker container"
	@echo ""
	@echo "📚 Dependency Management:"
	@echo "  make install_deps         - Install Python dependencies with Poetry"
	@echo ""
	@echo "📂 Data Loading and Processing:"
	@echo "  make load_data            - Load raw data from the database"
	@echo "  make download_images      - Download all product images"
	@echo "  make run_add_embeddings   - Add embeddings to the dataset using Python 3.8"
	@echo ""
	@echo "🤖 Machine Learning Pipeline:"
	@echo "  make train_mlflow         - Train a model with MLFlow monitoring"
	@echo "  make train                - Run the main machine learning training pipeline"
	@echo "  make image_clustering     - Perform clustering on images"
	@echo ""
	@echo "🗄️ Database and Clustering:"
	@echo "  make create_text_api_database - Create the text API database"
	@echo ""
	@echo "🧪 Testing and Cleanup:"
	@echo "  make run_tests            - Run unit and integration tests"
	@echo "  make clean                - Remove temporary and cache files"
	@echo ""
	@echo "❓ Help:"
	@echo "  make help                 - Display this help message"
