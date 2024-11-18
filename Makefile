# Variables
ML_CONFIG_PATH = training/config.yaml

# Build and run docker env
docker_up:
	@echo "Build and run docker env..."
	docker-compose up --build

# Stop containers
docker_down:
	@echo "Stop containers..."
	docker-compose down

# Visualize docker logs
docker_logs:
	docker-compose logs -f

# Install dependencies with Poetry
install_deps:
	@echo "Installing dependencies with Poetry..."
	poetry install
	poetry run python -m spacy download en_core_web_sm

# Run the main machine learning pipeline
run:
	@echo "Running the main pipeline..."
	poetry run python training/main.py --config $(ML_CONFIG_PATH)

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
	@echo "  make run_pipeline    - Run the main machine learning pipeline"
	@echo "  make clean           - Remove temporary and cache files"
	@echo "  make run_tests       - Run unit and integration tests"
