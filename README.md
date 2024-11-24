# 🍲 Foodstuffs Recommendation

A recommendation system designed to help you find similar food products based on a given item.

This tool enables consumers to choose alternative food substitutes to avoid allergens, opt for healthier options, or select more eco-friendly alternatives.

## 🧩 Project Components

This application consists of four main components:

- 🤖 **Training**: Scripts for training machine learning models.
- 📈 **MLflow**: Tracks and manages machine learning experiments and models.
- 🧑‍🍳 **API**: A FastAPI backend service to handle recommendations.
- 📊 **Dashboard**: A Streamlit-powered frontend for visualizing recommendations.

## 🚀 Getting Started

### Prerequisites

To run this project, ensure you have the following installed:

- **Docker**: [Download Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Download Docker Compose](https://docs.docker.com/compose/install/)
- **Poetry** (optional, for dependency management if developing locally): [Install Poetry](https://python-poetry.org/docs/)

### 🏗️ Project Structure

```
project-root/
├── api/                    # API service directory
│   ├── main.py             # Entrypoint for FastAPI
├── dashboard/              # Dashboard service directory
│   ├── streamlit_app.py    # Entrypoint for Streamlit
├── training/               # Model training scripts
│   ├── main.py             # Entrypoint for the ML pipeline
├── data/                   # Directory for datasets
│   └── raw/                # Raw data storage
│   └── processed/          # Processed data storage
│   └── production/         # Clean database for API and Dashboard
│   └── product_images/     # Product images storages for training with image
├── notebooks/              # Jupyter notebooks for exploratory analysis
├── scripts/                # Project scripts
├── docker/                 # Shared utilities and configuration
│   └── Dockerfile          # Dockerfile for the project
├── docker-compose.yml      # Docker Compose file for service orchestration
├── Makefile                # Makefile for automating tasks
└── README.md               # Project documentation
```

## 🛠️ Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/littlerobinson/foodstuffs-recommendation
   cd foodstuffs-recommendation
   ```

2. **Download data**:

   Place the dataset in the `data/raw` directory:

   ```bash
   wget https://static.openfoodfacts.org/data/raw/en.openfoodfacts.org.products.csv.gz -P data/raw
   ```

3. **Build and start the containers**:

   With Docker and Docker Compose installed, start the application by running:

   ```bash
   make docker_up
   ```

   This command will build and start the API, dashboard, and MLflow services as defined in the `docker-compose.yml` file.

4. **🌐 Accessing the Services**:

   - **API**: [http://localhost:8881](http://localhost:8881) 🧑‍🍳
   - **Dashboard**: [http://localhost:8882](http://localhost:8882) 📊
   - **MLflow**: [http://localhost:8883](http://localhost:8883) 📈

## 🔧 Development and Model Training

- **Model Training**: Run training scripts located in the `training` directory.
- **Datasets**: Place any required datasets in the `data` directory.
- **Exploratory Analysis**: Jupyter notebooks for research are available in the `notebooks` directory.

### Using the Makefile

### Using the Makefile

To automate common tasks, you can use the Makefile. Below is the list of commands in the correct execution order and with comments on their purpose:

   - `make docker_up`: Builds and starts the Docker containers for the API, dashboard, and MLflow.
   - `make docker_down`: Stops and removes the running Docker containers.
   - `make docker_logs`: Displays live logs from the running Docker containers for debugging.
   - `make export_secrets`: Exports secrets from the `secrets.sh` file by making it executable.
   
   *Run the Docker CLI for Python 3.8*

   - `make run_docker_cli`: Opens an interactive terminal session within the Python 3.12 Docker container.
   - `make install_deps`: Installs all project dependencies and downloads SpaCy's `en_core_web_sm` model.
   - `make load_data`: Processes and loads raw data into the database, as specified in `training/config.yaml`.
   - `make mlflow`: Starts MLFlow for tracking experiments and model performance.
   - `make run`: Combines the `load_data` and `mlflow` commands to execute the full training pipeline.
   - `make download_images`: Downloads images needed for the project based on the configuration file.

   *Run the Docker CLI for Python 3.8*
   - `make run_python_cli_3_8`: Opens an interactive terminal session within the Python 3.8 Docker container.
   - `make run_add_embeddings`: Generates embeddings from image files and adds them to the dataset. (Requires Python 3.8)

   *On the Docker CLI for Python 3.12*
   - `make run_image_clustering`: Clusters images using embeddings, as specified in the configuration file.

   - `make clean`: Removes temporary files, caches, and Python bytecode.

   - `make run_tests`: Executes all unit and integration tests for both the API and training modules.


### Example

If you want to start the services and train the model in one go, you can use:

```bash
make docker_up && make run
```

## 🚀 Usage

After starting the services, you can:

- **API**: Make POST requests with product data to receive similar product recommendations.
- **Dashboard**: Interactively explore and visualize product recommendations.
- **MLflow**: Track metrics, hyperparameters, and model performance.

Enjoy discovering new products and exploring healthier, allergen-free, or eco-friendly alternatives! 🥳

## 🚀 Deployment exemple on Heroku

```bash
# Build the project images
docker compose build

#  Connection to Heroku and the container registry
heroku login
heroku container:login

# Tag image for Heroku register
docker tag <container-name> registry.heroku.com/<container-name>/web

#  Push image to Heroku
docker push registry.heroku.com/<container-name>/web

# Deploying the image
heroku container:release web -a <container-name>

```

## ➕ Advanced Features

1. **Sustainability Scoring**: Combine factors such as packaging, CO2 emissions, and product origins to create a custom environmental score. You can then analyze this score’s correlation with product categories and processing levels.
2. **Nutri-Score Prediction**: Use product attributes to train a model that predicts a product’s Nutri-Score or eco-score.
