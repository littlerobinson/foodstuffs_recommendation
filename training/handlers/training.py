import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from utils.logger import setup_logger

from handlers.data_params import CATEGORIAL_COLUMNS, NUMERIC_COLUMNS

logger = setup_logger()


class Training:
    def __init__(self, encoding_method_name="svd"):
        """
        Initializes the DataTraining object.
        """
        if encoding_method_name == "svd":
            self.encoding_method = self.encode_categorical_data_with_svd
        elif encoding_method_name == "pca":
            self.encoding_method = self.encode_categorical_data_with_pca
        else:
            raise ValueError("Encoding method is unknow")
        logger.info(f"Instantiating DataTraining object with {encoding_method_name}.")

    def __impute_numeric_data(self, df, columns):
        """
        Imputes missing values in the specified numeric columns using the mean of each column.

        Parameters:
            df (DataFrame): The DataFrame containing the data to impute.
            columns (list): List of numeric columns to impute.

        Returns:
            DataFrame: The DataFrame with missing values imputed in the specified columns.
        """
        imputer = SimpleImputer(strategy="mean")
        df[columns] = imputer.fit_transform(df[columns])
        return df

    def __scale_numeric_data(self, df, columns):
        """
        Scales the specified numeric columns using standardization (z-score normalization).

        Parameters:
            df (DataFrame): The DataFrame containing the data to scale.
            columns (list): List of numeric columns to scale.

        Returns:
            DataFrame: The DataFrame with scaled values in the specified columns.
        """
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df

    def encode_categorical_data_with_svd(
        self, df, columns, max_features=500, min_df=4, n_components=50
    ):
        """
        Encodes categorical text data using CountVectorizer followed by dimensionality reduction with TruncatedSVD.

        Parameters:
            df (DataFrame): The DataFrame containing the data.
            columns (list): List of categorical columns to encode.
            max_features (int): Maximum number of features for CountVectorizer.
            min_df (int): Minimum document frequency for CountVectorizer.
            n_components (int): Number of components for SVD.

        Returns:
            DataFrame: A DataFrame with encoded and dimensionally-reduced categorical features.
        """
        logger.info("Encoding categorical data with SVD.")
        vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)
        encoded_columns = []

        for col in columns:
            logger.info(f"Vectorization of column: {col}")
            # Encode text data
            encoded = vectorizer.fit_transform(df[col].fillna(""))
            # Adjust number of components for SVD if needed
            adjusted_n_components = min(n_components, encoded.shape[1])
            svd = TruncatedSVD(n_components=adjusted_n_components, random_state=42)
            # Reduce dimensions
            reduced = svd.fit_transform(encoded)
            encoded_columns.append(pd.DataFrame(reduced))

        return pd.concat(encoded_columns, axis=1)

    def encode_categorical_data_with_pca(
        self, df, columns, max_features=500, min_df=4, n_components=50
    ):
        """
        Encodes categorical text data using CountVectorizer followed by dimensionality reduction with PCA.

        Parameters:
            df (DataFrame): The DataFrame containing the data.
            columns (list): List of categorical columns to encode.
            max_features (int): Maximum number of features for CountVectorizer.
            min_df (int): Minimum document frequency for CountVectorizer.
            n_components (int): Number of components for PCA.

        Returns:
            DataFrame: A DataFrame with encoded and dimensionally-reduced categorical features.
        """
        logger.info("Encoding categorical data with PCA.")
        vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)
        encoded_columns = []

        for col in columns:
            logger.info(f"Vectorization of column: {col}")
            # Encode text data
            encoded = vectorizer.fit_transform(df[col].fillna(""))
            # Convert to dense matrix for PCA
            dense_encoded = encoded.toarray()
            # Adjust number of components for PCA if needed
            adjusted_n_components = min(n_components, dense_encoded.shape[1])
            pca = PCA(n_components=adjusted_n_components, random_state=42)
            # Reduce dimensions
            reduced = pca.fit_transform(dense_encoded)
            encoded_columns.append(pd.DataFrame(reduced))

        return pd.concat(encoded_columns, axis=1)

    def train_kmeans(
        self,
        df,
        n_clusters,
        random_state=42,
    ):
        """
        Applies KMeans clustering with a specified number of clusters.

        Parameters:
            df (DataFrame): The DataFrame containing the data.
            n_clusters (int): The number of clusters for KMeans.
            encoding_method (function): Method to encode categorical columns.
            random_state (int): Random state for reproducibility.

        Returns:
            kmeans: The kmeans model
            metrics: dic of metrics
            labels: dataframe with cluster column
        """
        df = self.__impute_numeric_data(df, NUMERIC_COLUMNS)
        df = self.__scale_numeric_data(df, NUMERIC_COLUMNS)
        categorical_features = self.encoding_method(df, CATEGORIAL_COLUMNS)
        categorical_features.to_csv(
            "data/production/categorical_features.csv", index=False
        )
        features = pd.concat(
            [categorical_features, df[NUMERIC_COLUMNS].reset_index(drop=True)], axis=1
        )
        # Converting column names to strings
        features.columns = features.columns.astype(str)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(features)

        inertia = kmeans.inertia_
        silhouette_metric = silhouette_score(features, labels)
        davies_bouldin_metric = davies_bouldin_score(features, labels)
        calinski_harabasz_metric = calinski_harabasz_score(features, labels)

        metrics = {
            "kmeans_inertia": inertia,
            "silhouette_metric": silhouette_metric,
            "davies_bouldin_metric": davies_bouldin_metric,
            "calinski_harabasz_metric": calinski_harabasz_metric,
        }

        return kmeans, features, metrics, labels

    def train_dbscan(self, df, eps=0.5, min_samples=5, metric="euclidean"):
        """
        Applies DBSCAN clustering on encoded and normalized data.

        Parameters:
            df (DataFrame): The DataFrame containing the data.
            encoding_method (function): Method to encode categorical columns.
            eps (float): Maximum distance between points for them to be in the same cluster.
            min_samples (int): Minimum points required to form a cluster.
            metric (str): Distance metric for DBSCAN.

        Returns:
            dbscan: The DBSCAN model
            metrics: dic of metrics
            labels: dataframe with cluster column
        """
        df = self.__impute_numeric_data(df, NUMERIC_COLUMNS)
        df = self.__scale_numeric_data(df, NUMERIC_COLUMNS)
        categorical_features = self.encoding_method(df, CATEGORIAL_COLUMNS)
        features = pd.concat(
            [categorical_features, df[NUMERIC_COLUMNS].reset_index(drop=True)], axis=1
        )
        # Converting column names to strings
        features.columns = features.columns.astype(str)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = dbscan.fit_predict(features)

        silhouette_metric = (
            silhouette_score(features, labels) if len(set(labels)) > 1 else None
        )
        davies_bouldin_metric = (
            davies_bouldin_score(features, labels) if len(set(labels)) > 1 else None
        )
        calinski_harabasz_metric = (
            calinski_harabasz_score(features, labels) if len(set(labels)) > 1 else None
        )

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        metrics = {
            "silhouette_metric": silhouette_metric,
            "davies_bouldin_metric": davies_bouldin_metric,
            "calinski_harabasz_metric": calinski_harabasz_metric,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }

        return dbscan, features, metrics, labels
