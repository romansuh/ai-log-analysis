from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class LogClusterVisualizer:
    def __init__(self, n_clusters=5):
        """
        Initialize the visualizer with the number of clusters to create.
        
        Args:
            n_clusters (int): Number of clusters for k-means
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
    def fit_predict(self, embeddings):
        """
        Fit KMeans and return cluster labels based on BERT vector similarity.
        
        Args:
            embeddings: BERT vectors from the model
            
        Returns:
            Cluster labels for each vector
        """
        # Normalize vectors to ensure cosine similarity
        normalized_vectors = normalize(embeddings)
        return self.kmeans.fit_predict(normalized_vectors)
        
    def visualize_clusters(self, embeddings):
        """
        Visualize clusters of log messages using BERT embeddings.
        Uses PCA to reduce dimensionality for visualization.
        """
        # Get cluster labels using normalized vectors
        clusters = self.fit_predict(embeddings)
        
        # Reduce dimensionality to 2D using PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(normalize(embeddings))
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        
        # Add some sample messages as annotations (every 20th point to avoid overcrowding)
        for i in range(0, len(reduced), 200):
            plt.annotate(f'Msg_{i}', (reduced[i, 0], reduced[i, 1]))
        
        plt.title('Log Message Clusters (KMeans on BERT embeddings)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.show()
        
        return clusters

def add_kmeans_clusters(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    Add KMeans cluster labels to the DataFrame based on BERT vector similarity.
    
    Args:
        df: DataFrame containing BERT vectors
        n_clusters: Number of clusters to create
        
    Returns:
        DataFrame with added KMeans_cluster_id column
    """
    # Validate input
    if 'bert_vector' not in df.columns:
        raise ValueError("DataFrame must contain 'bert_vector' column")
    
    # Extract BERT vectors
    vectors = np.vstack(df['bert_vector'].values)
    
    # Create visualizer and get clusters
    visualizer = LogClusterVisualizer(n_clusters=n_clusters)
    clusters = visualizer.fit_predict(vectors)
    
    # Add clusters to DataFrame
    df = df.copy()
    df['KMeans_cluster_id'] = clusters
    
    return df