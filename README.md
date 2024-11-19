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
│   ├── main.py             # Entrypoint for Streamlit
├── training/               # Model training scripts
│   ├── main.py             # Entrypoint for the ML pipeline
│   ├── data/               # Data loading and preprocessing
│   └── models/             # Model training and evaluation logic
├── data/                   # Directory for datasets
│   └── raw/                # Raw data storage
│   └── processed/          # Processed data storage
├── notebooks/              # Jupyter notebooks for exploratory analysis
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

To automate common tasks, you can use the Makefile. Here are the available commands:

- `make docker_up`: Build and start the Docker containers for API, dashboard, and MLflow.
- `make docker_down`: Stop and remove the Docker containers.
- `make run`: Run the model training pipeline (starts the training process for the machine learning model).
- `make install`: Install dependencies via Poetry.

### Example

If you want to start the services and train the model in one go, you can use:

```bash
make docker_up && make train
```

## 🚀 Usage

After starting the services, you can:

- **API**: Make POST requests with product data to receive similar product recommendations.
- **Dashboard**: Interactively explore and visualize product recommendations.
- **MLflow**: Track metrics, hyperparameters, and model performance.

Enjoy discovering new products and exploring healthier, allergen-free, or eco-friendly alternatives! 🥳

## ➕ Advanced Features

1. **Sustainability Scoring**: Combine factors such as packaging, CO2 emissions, and product origins to create a custom environmental score. You can then analyze this score’s correlation with product categories and processing levels.
2. **Nutri-Score Prediction**: Use product attributes to train a model that predicts a product’s Nutri-Score or eco-score.
