from utils.logger import setup_logger

from handlers.training import Training

logger = setup_logger()


def build_pipeline(data, n_clusters, encoding_method_name):
    logger.info("Build training pipeline, work in progress 🚧.")

    # Launch training pipeline
    training = Training(encoding_method_name=encoding_method_name)

    model, features, metrics, labels = training.train_kmeans(
        df=data, n_clusters=n_clusters
    )
    # model, features, metrics, labels = training.train_dbscan(df=data)

    return model, features, metrics, labels
