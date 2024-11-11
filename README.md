# 🍲 Foodstuffs Recommendation

A recommendation system designed to help you find similar food products based on a given item.

Enable consumers to choose alternative food substitutes to avoid food allergies, for example, or healthier or more environmentally-friendly alternatives.

This application consists of three main components:

- 🤖 **Training**: Machine learning training models script.
- 📈 **MLflow**: Tracks the machine learning models and experiments.
- 🧑‍🍳 **API**: The backend service to access to predictions make with FastAPI.
- 📊 **Dashboard**: A Streamlit-powered frontend for visualizing product recommendations make withe Streamlit.

## 🚀 Getting Started

### Prerequisites

To run this project, ensure you have the following installed:

- **Docker**: [Download Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Download Docker Compose](https://docs.docker.com/compose/install/)

### 🏗️ Project Structure

```
project-root
├── api/                    # API service directory
│   └── src/                # API source code
├── dashboard/              # Dashboard (Streamlit) service directory
│   └── src/                # Dashboard source code
├── docker/                 # Dockerfiles for each service
│   └── Dockerfile          # Multi-stage Dockerfile for building each component
├── training/               # Model training scripts
│   └── train_model.py      # Model training script
├── data/                   # Directory for datasets
│   └── dataset.csv         # Example dataset file
├── notebooks/              # Jupyter notebooks for exploratory analysis
│   └── exploratory_analysis.ipynb # Example notebook for research
├── docker-compose.yml      # Docker Compose file to orchestrate the services
└── README.md               # Project documentation
```

## 🛠️ Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/littlerobinson/foodstuffs-recommendation
   cd foodstuffs-recommendation
   ```

2. **Build and start the containers**:

After installing Docker and Docker Compose, you can start the application by running the following command in the project root:

```bash
docker-compose up --build
```

This command will build and launch the API, dashboard, and MLflow services as defined in the `docker-compose.yml` file.

3. **🌐 Accessing the Services**:

   - **API**: [http://localhost:8881](http://localhost:8881) 🧑‍🍳
   - **Dashboard**: [http://localhost:8882](http://localhost:8882) 📊
   - **MLflow**: [http://localhost:8883](http://localhost:8883) 📈

### 🔧 Development and Model Training

- **Model Training Scripts**: Located in the `training` directory.
- **Datasets**: Place datasets in the `data` directory.
- **Exploratory Analysis**: Use Jupyter notebooks located in the `notebooks` directory for research and exploration.

## 🚀 Usage

After starting the services, you can explore the following:

- **API**: Make POST requests with product data to receive similar product recommendations.
- **Dashboard**: Interactively explore and visualize product recommendations.
- **MLflow**: Track experiment metrics, parameters, and model performance.

Enjoy exploring similar products and finding new favorites products! 🥳

## ➕ Bonus

1. Create a composite sustainability score based on packaging, CO2 emissions, and geographical origin of products, weighting each environmental factor according to its impact, and test the score by correlating it with product categories or processing levels.
2. Use existing product data to train a machine learning model to predict a product's Nutri-Score or ecological score.
