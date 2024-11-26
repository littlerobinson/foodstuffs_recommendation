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

### 🧪 Makefile Documentation

This Makefile provides commands to manage the Docker environment, run scripts, load data, and handle project dependencies.

---

#### Variables

- **`ML_CONFIG_PATH`**: Path to the YAML configuration file for training (`training/config.yaml`).
- **`SECRET_FILE`**: Path to the secrets file (`secrets.sh`).

---

#### Commands

##### Docker Management

- **`make docker_up`**  
  Build and start the Docker environment.  
  **Example:**

  ```bash
  make docker_up
  ```

- **`make docker_down`**  
  Stop all running Docker containers.  
  **Example:**

  ```bash
  make docker_down
  ```

- **`make docker_logs`**  
  View real-time logs from Docker containers.  
  **Example:**
  ```bash
  make docker_logs
  ```

---

##### Secrets Management

- **`make export_secrets`**  
  Load and export environment secrets from `secrets.sh`.  
  **Example:**
  ```bash
  make export_secrets
  ```

---

##### Command Line Access

- **`make run_docker_cli`**  
  Open a shell in the Docker container for the Python CLI.  
  **Example:**
  ```bash
  make run_docker_cli
  ```

---

##### Dependency Management

- **`make install_deps`**  
  Install dependencies using Poetry and download the `en_core_web_sm` spaCy model.  
  **Example:**
  ```bash
  make install_deps
  ```

---

##### Data Loading and Processing

- **`make load_data`**  
  Load raw data from the database using the configuration file.  
  **Example:**

  ```bash
  make load_data
  ```

- **`make download_images`**  
  Download all product images based on the configuration file.  
  **Example:**

  ```bash
  make download_images
  ```

- **`make run_add_embeddings`**  
  Add embeddings to the dataset using Python 3.8.  
  **Example:**
  ```bash
  make run_add_embeddings
  ```

---

##### Machine Learning Pipeline and clustering

- **`make train_mlflow`**  
  Run training with MLFlow monitoring.  
  **Example:**

  ```bash
  make train_mlflow
  ```

- **`make train`**  
  Execute the main machine learning pipeline, including data loading and MLFlow initialization.  
  **Example:**

  ```bash
  make train
  ```

- **`make image_clustering`**  
  Perform clustering on images using the specified configuration.  
  **Example:**
  ```bash
  make image_clustering
  ```

---

##### Database

- **`make create_text_api_database`**  
  Create the text database for the API.  
  **Example:**

  ```bash
  make create_text_api_database
  ```

---

##### Testing and Cleanup

- **`make run_tests`**  
  Run all unit and integration tests in the project.  
  **Example:**

  ```bash
  make run_tests
  ```

- **`make clean`**  
  Remove temporary files and cache directories.  
  **Example:**
  ```bash
  make clean
  ```

---

##### Help

- **`make help`**  
  Display a list of available Makefile commands.  
  **Example:**
  ```bash
  make help
  ```

---

##### Example

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
# Create Heroku project
heroku create --region eu <container-name>

# Set as docker app
heroku stack:set container -a <container-name>

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
