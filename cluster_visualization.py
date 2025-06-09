from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class LogClusterVisualizer:
    def __init__(self, n_clusters=5):
        """
        Initialize the visualizer with the number of clusters to create.
        
        Args:
            n_clusters (int): Number of clusters for k-means
        """
        self.n_clusters = n_clusters
        
    def visualize_clusters(self, embeddings):
        """Visualize clusters of log messages using BERT embeddings."""
        # Reduce dimensionality to 2D using PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        
        # Add some sample messages as annotations (every 20th point to avoid overcrowding)
        for i in range(0, len(reduced), 20):
            plt.annotate(f'Msg_{i}', (reduced[i, 0], reduced[i, 1]))
        
        plt.title('Log Message Clusters (KMeans on BERT embeddings)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.show()
        
        return clusters