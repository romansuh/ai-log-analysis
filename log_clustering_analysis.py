#!/usr/bin/env python3
"""
Log Clustering Analysis using BERT Embeddings
==============================================

This script extracts BERT embeddings from log messages, applies K-Means clustering,
and visualizes the results using PCA and t-SNE dimensionality reduction.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

from src.bert_pipeline import BERTConfig, LogBERTPipeline

class LogEmbeddingExtractor:
    """Extract BERT embeddings from log messages."""
    
    def __init__(self, model_name="distilbert-base-uncased", max_length=128):
        """
        Initialize the embedding extractor.
        
        Args:
            model_name: BERT model to use
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
    def extract_embeddings(self, texts, batch_size=16):
        """
        Extract BERT embeddings from texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings [num_texts, embedding_dim]
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                encoded = self.tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Get BERT embeddings
                outputs = self.model(**encoded)
                
                # Use [CLS] token embedding (first token) as sentence representation
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
                embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)


def generate_diverse_log_dataset(size=300):
    """Generate a diverse dataset of log messages for clustering analysis."""
    
    print(f"📊 Generating diverse log dataset with {size} entries...")
    
    # Define log categories with distinct patterns
    log_categories = {
        'authentication': [
            "User {user} successfully logged in from IP {ip}",
            "Failed login attempt for user {user} from {ip}",
            "Password reset requested for user {user}",
            "User {user} session expired after {duration}",
            "Two-factor authentication enabled for user {user}",
        ],
        'database': [
            "Database query executed in {duration}: SELECT * FROM {table}",
            "Database connection pool exhausted, max {count} connections",
            "Slow query detected: {query} took {duration}",
            "Database backup completed: {size} written to {path}",
            "Connection timeout to database {db} after {duration}",
        ],
        'api_requests': [
            "GET {endpoint} returned {status} in {duration}",
            "POST request to {endpoint} with {size} payload",
            "API rate limit exceeded for client {client}: {count} requests",
            "Invalid API key used for endpoint {endpoint}",
            "API response cached for {endpoint}, expires in {duration}",
        ],
        'system_resources': [
            "High CPU usage detected: {percentage}% on core {core}",
            "Memory usage critical: {used}/{total} GB used",
            "Disk space warning: {percentage}% full on {partition}",
            "Network bandwidth exceeded: {speed} Mbps on interface {interface}",
            "Temperature alert: CPU at {temp}°C, threshold {threshold}°C",
        ],
        'security': [
            "Suspicious file access: {file} by process {pid}",
            "Port scan detected from IP {ip} targeting ports {ports}",
            "Malware signature found in file {file}",
            "Firewall blocked connection from {ip} to port {port}",
            "SSL certificate expired for domain {domain}",
        ],
        'application_errors': [
            "Application crashed with exit code {code}, PID {pid}",
            "Null pointer exception in method {method} at line {line}",
            "OutOfMemoryError: Java heap space in {component}",
            "Thread deadlock detected between {thread1} and {thread2}",
            "Configuration file {file} not found, using defaults",
        ]
    }
    
    # Sample data for templates
    sample_data = {
        'user': ['admin', 'john.doe', 'alice.smith', 'bob.jones', 'api_service'],
        'ip': ['192.168.1.100', '10.0.0.50', '172.16.0.1', '203.0.113.1'],
        'duration': ['30s', '245ms', '1.2hr', '5min', '500ms'],
        'table': ['users', 'logs', 'sessions', 'products'],
        'count': ['15', '100', '50', '500'],
        'endpoint': ['/api/users', '/api/data', '/health', '/login'],
        'status': ['200', '404', '500', '401', '403'],
        'size': ['2.1KB', '512MB', '1.2GB', '15MB'],
        'client': ['mobile_app', 'web_client', 'admin_panel'],
        'percentage': ['85', '92', '78', '95'],
        'core': ['0', '1', '2', '3'],
        'total': ['8', '16', '32', '64'],
        'used': ['6.8', '14.2', '28.5', '58.1'],
        'partition': ['/var', '/home', '/tmp', '/opt'],
        'speed': ['100', '250', '500', '1000'],
        'interface': ['eth0', 'wlan0', 'docker0'],
        'temp': ['75', '82', '90', '78'],
        'threshold': ['80', '85', '90'],
        'file': ['/etc/passwd', '/var/log/app.log', '/tmp/upload.zip'],
        'pid': ['1234', '5678', '9012'],
        'ports': ['22,80,443', '3306,5432', '8080,9000'],
        'port': ['22', '80', '443', '3306'],
        'domain': ['example.com', 'api.service.com', 'internal.local'],
        'code': ['0', '1', '-1', '137'],
        'method': ['getUserData()', 'processRequest()', 'validateInput()'],
        'line': ['142', '67', '89', '203'],
        'component': ['UserService', 'DataProcessor', 'CacheManager'],
        'thread1': ['worker-1', 'scheduler', 'http-handler'],
        'thread2': ['db-pool', 'cache-writer', 'log-processor'],
        'query': ['SELECT COUNT(*)', 'UPDATE users SET', 'DELETE FROM temp'],
        'db': ['userdb', 'logdb', 'analytics'],
        'path': ['/backup/daily.sql', '/archive/logs.gz']
    }
    
    logs = []
    categories = []
    
    for _ in range(size):
        # Choose category
        category = np.random.choice(list(log_categories.keys()))
        template = np.random.choice(log_categories[category])
        
        # Fill template
        filled_template = template
        for key, values in sample_data.items():
            if '{' + key + '}' in template:
                filled_template = filled_template.replace('{' + key + '}', np.random.choice(values))
        
        # Add timestamp and log level
        timestamp = f"2024-12-24 {np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}:{np.random.randint(0, 60):02d}"
        level = np.random.choice(['INFO', 'WARNING', 'ERROR', 'DEBUG'], p=[0.5, 0.25, 0.15, 0.1])
        
        log_entry = f"{timestamp} {level} {filled_template}"
        
        logs.append(log_entry)
        categories.append(category)
    
    df = pd.DataFrame({
        'log_message': logs,
        'true_category': categories,
        'message_length': [len(msg) for msg in logs]
    })
    
    print(f"✅ Generated {len(df)} log entries")
    print(f"📋 Categories: {df['true_category'].value_counts().to_dict()}")
    
    return df


def perform_clustering_analysis(embeddings, texts, true_categories, n_clusters_range=(2, 10)):
    """
    Perform K-Means clustering analysis with different numbers of clusters.
    
    Args:
        embeddings: BERT embeddings array
        texts: Original log texts
        true_categories: Ground truth categories
        n_clusters_range: Range of cluster numbers to try
        
    Returns:
        Dictionary with clustering results
    """
    
    print(f"\n🔍 CLUSTERING ANALYSIS")
    print("=" * 30)
    
    results = {}
    silhouette_scores = []
    ch_scores = []
    cluster_range = range(n_clusters_range[0], n_clusters_range[1] + 1)
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        sil_score = silhouette_score(embeddings, cluster_labels)
        ch_score = calinski_harabasz_score(embeddings, cluster_labels)
        
        silhouette_scores.append(sil_score)
        ch_scores.append(ch_score)
        
        print(f"  k={n_clusters}: Silhouette={sil_score:.3f}, CH Score={ch_score:.1f}")
    
    # Find optimal k using silhouette score
    optimal_k = cluster_range[np.argmax(silhouette_scores)]
    
    print(f"\n🎯 Optimal number of clusters: {optimal_k}")
    print(f"   Best silhouette score: {max(silhouette_scores):.3f}")
    
    # Perform final clustering with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(embeddings)
    
    results = {
        'optimal_k': optimal_k,
        'cluster_labels': final_labels,
        'silhouette_scores': silhouette_scores,
        'ch_scores': ch_scores,
        'cluster_range': list(cluster_range),
        'kmeans_model': final_kmeans,
        'cluster_centers': final_kmeans.cluster_centers_
    }
    
    return results


def visualize_clusters(embeddings, cluster_labels, true_categories, texts, results):
    """
    Create comprehensive visualizations of the clustering results.
    
    Args:
        embeddings: BERT embeddings
        cluster_labels: Predicted cluster labels
        true_categories: Ground truth categories
        texts: Original log texts
        results: Clustering analysis results
    """
    
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("=" * 30)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Cluster evaluation metrics
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(results['cluster_range'], results['silhouette_scores'], 'bo-', label='Silhouette Score')
    plt.axvline(x=results['optimal_k'], color='red', linestyle='--', alpha=0.7, label=f'Optimal k={results["optimal_k"]}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Cluster Evaluation: Silhouette Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Calinski-Harabasz score
    ax2 = plt.subplot(3, 3, 2)
    plt.plot(results['cluster_range'], results['ch_scores'], 'go-', label='CH Score')
    plt.axvline(x=results['optimal_k'], color='red', linestyle='--', alpha=0.7, label=f'Optimal k={results["optimal_k"]}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Cluster Evaluation: CH Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. PCA visualization
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    ax3 = plt.subplot(3, 3, 4)
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('K-Means Clusters (PCA)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    
    # 4. PCA with true categories
    ax4 = plt.subplot(3, 3, 5)
    unique_categories = list(set(true_categories))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
    
    for i, category in enumerate(unique_categories):
        mask = np.array(true_categories) == category
        plt.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                   c=[colors[i]], label=category, alpha=0.7, s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('True Categories (PCA)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 5. t-SNE visualization
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    ax5 = plt.subplot(3, 3, 7)
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('K-Means Clusters (t-SNE)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    
    # 6. t-SNE with true categories
    ax6 = plt.subplot(3, 3, 8)
    for i, category in enumerate(unique_categories):
        mask = np.array(true_categories) == category
        plt.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
                   c=[colors[i]], label=category, alpha=0.7, s=50)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('True Categories (t-SNE)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 7. Cluster size distribution
    ax7 = plt.subplot(3, 3, 3)
    cluster_counts = np.bincount(cluster_labels)
    plt.bar(range(len(cluster_counts)), cluster_counts, alpha=0.7)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Log Messages')
    plt.title('Cluster Size Distribution')
    plt.grid(True, alpha=0.3)
    
    # 8. Confusion matrix heatmap (if we can map clusters to categories)
    ax8 = plt.subplot(3, 3, 6)
    
    # Create confusion matrix between clusters and true categories
    from sklearn.metrics import adjusted_rand_score
    
    # Calculate cluster-category mapping
    cluster_category_matrix = pd.crosstab(
        pd.Series(cluster_labels, name='Cluster'),
        pd.Series(true_categories, name='True Category')
    )
    
    sns.heatmap(cluster_category_matrix, annot=True, fmt='d', cmap='Blues', ax=ax8)
    plt.title('Cluster vs True Category')
    plt.ylabel('Predicted Cluster')
    plt.xlabel('True Category')
    
    # 9. Sample log messages from each cluster
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    sample_text = "Sample Log Messages by Cluster:\n\n"
    for cluster_id in range(results['optimal_k']):
        cluster_mask = cluster_labels == cluster_id
        cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
        
        if len(cluster_texts) > 0:
            sample_text += f"Cluster {cluster_id} ({sum(cluster_mask)} logs):\n"
            # Show first 2 examples (truncated)
            for i, text in enumerate(cluster_texts[:2]):
                truncated = text[:80] + "..." if len(text) > 80 else text
                sample_text += f"  • {truncated}\n"
            sample_text += "\n"
    
    ax9.text(0.02, 0.98, sample_text, transform=ax9.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('log_clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and print clustering metrics
    ari_score = adjusted_rand_score(true_categories, cluster_labels)
    print(f"\n📈 CLUSTERING METRICS:")
    print(f"   Adjusted Rand Index: {ari_score:.3f}")
    print(f"   Silhouette Score: {max(results['silhouette_scores']):.3f}")
    print(f"   Number of clusters: {results['optimal_k']}")


def main():
    """Main analysis function."""
    
    print("🔍 LOG CLUSTERING ANALYSIS WITH BERT EMBEDDINGS")
    print("=" * 60)
    
    # Generate diverse log dataset
    df = generate_diverse_log_dataset(300)
    
    # Initialize BERT pipeline for preprocessing
    print(f"\n🔧 INITIALIZING BERT PIPELINE")
    print("-" * 30)
    
    config = BERTConfig(
        model_name="distilbert-base-uncased",
        max_length=128,
        batch_size=16
    )
    
    pipeline = LogBERTPipeline(config)
    pipeline.fit(df['log_message'])
    
    # Preprocess logs
    print("Preprocessing log messages...")
    preprocessed_result = pipeline.transform(df['log_message'])
    
    # Extract BERT embeddings
    print(f"\n🤖 EXTRACTING BERT EMBEDDINGS")
    print("-" * 30)
    
    embedder = LogEmbeddingExtractor(
        model_name="distilbert-base-uncased",
        max_length=128
    )
    
    print("Extracting embeddings from preprocessed logs...")
    # Use preprocessed texts for embedding extraction
    processed_texts = []
    for i in range(len(df)):
        # Get the preprocessed text by decoding tokens and removing special tokens
        tokens = preprocessed_result['input_ids'][i]
        processed_text = embedder.tokenizer.decode(tokens, skip_special_tokens=True)
        processed_texts.append(processed_text)
    
    embeddings = embedder.extract_embeddings(processed_texts, batch_size=16)
    
    print(f"✅ Extracted embeddings shape: {embeddings.shape}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    # Perform clustering analysis
    clustering_results = perform_clustering_analysis(
        embeddings, 
        df['log_message'].tolist(),
        df['true_category'].tolist(),
        n_clusters_range=(3, 8)
    )
    
    # Create visualizations
    visualize_clusters(
        embeddings,
        clustering_results['cluster_labels'],
        df['true_category'].tolist(),
        df['log_message'].tolist(),
        clustering_results
    )
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"   Visualization saved as 'log_clustering_analysis.png'")
    print(f"   Found {clustering_results['optimal_k']} optimal clusters")


if __name__ == "__main__":
    main() 