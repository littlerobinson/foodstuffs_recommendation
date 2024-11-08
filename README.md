# 🍽️ Foodstuffs Recommendation App

A recommendation system designed to help you find similar food products based on a given item. This application consists of three main components:

- 🤖 **Training**: Machine learning training models script.
- 📈 **MLflow**: Tracks the machine learning models and experiments.
- 🧑‍🍳 **API**: The backend service to access to predictions make with FastAPI.
- 📊 **Dashboard**: A Streamlit-powered frontend for visualizing product recommendations make withe Streamlit.

## 🗂️ Project Structure

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

   ```bash
   docker-compose up --build
   ```

3. **Access the services**:

   - **API**: [http://localhost:8881](http://localhost:8881) 🧑‍🍳
   - **Dashboard**: [http://localhost:8882](http://localhost:8882) 📊
   - **MLflow**: [http://localhost:8883](http://localhost:8883) 📈

## 🐳 Docker Setup

Each service is defined in the Docker Compose file:

- **API service**:

  - **Container name**: `foodstuffs-recommendation-api`
  - **Port**: `8881:8881`
  - **Code directory**: `./api/src` (mounted to `/app/src` inside the container)

- **Dashboard service**:

  - **Container name**: `foodstuffs-recommendation-dashboard`
  - **Port**: `8882:8882`
  - **Code directory**: `./dashboard/src` (mounted to `/app/src` inside the container)

- **MLflow service**:
  - **Container name**: `foodstuffs-recommendation-mlflow`
  - **Port**: `8883:8883`

## 🚀 Usage

After starting the services, you can explore the following:

- **API**: Make POST requests with product data to receive similar product recommendations.
- **Dashboard**: Interactively explore and visualize product recommendations.
- **MLflow**: Track experiment metrics, parameters, and model performance.

Enjoy exploring similar products and finding new favorites! 🥳
