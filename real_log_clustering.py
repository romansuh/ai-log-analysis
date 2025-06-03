#!/usr/bin/env python3
"""
Real Log Clustering Analysis using BERT Embeddings
==================================================

This script processes real Zookeeper logs, extracts BERT embeddings, 
applies K-Means clustering, and visualizes patterns with PCA and t-SNE.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from transformers import AutoTokenizer, AutoModel
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from src.bert_pipeline import BERTConfig, LogBERTPipeline

class ZookeeperLogParser:
    """Parse and analyze Zookeeper log files."""
    
    def __init__(self):
        self.log_patterns = {}
        self.message_templates = {}
        
    def parse_log_file(self, filepath, sample_size=None):
        """
        Parse Zookeeper log file and extract structured information.
        
        Args:
            filepath: Path to log file
            sample_size: Optional sample size for large files
            
        Returns:
            DataFrame with parsed log data
        """
        print(f"📁 Parsing log file: {filepath}")
        
        logs = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                    
                parsed = self._parse_log_line(line.strip())
                if parsed:
                    logs.append(parsed)
        
        df = pd.DataFrame(logs)
        print(f"✅ Parsed {len(df)} log entries")
        
        # Add pattern analysis
        df = self._add_pattern_analysis(df)
        
        return df
    
    def _parse_log_line(self, line):
        """Parse a single log line."""
        # Zookeeper log format: YYYY-MM-DD HH:MM:SS,mmm - LEVEL [context] - message
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+)\s+\[([^\]]+)\] - (.+)'
        
        match = re.match(pattern, line)
        if match:
            timestamp, level, context, message = match.groups()
            return {
                'timestamp': timestamp,
                'log_level': level,
                'context': context,
                'message': message,
                'full_log': line,
                'message_length': len(message)
            }
        return None
    
    def _add_pattern_analysis(self, df):
        """Add pattern analysis to identify log message types."""
        print("🔍 Analyzing log patterns...")
        
        # Extract patterns by removing variable content
        patterns = []
        for message in df['message']:
            # Replace common variable patterns
            pattern = message
            
            # Replace IP addresses
            pattern = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', pattern)
            
            # Replace ports
            pattern = re.sub(r':\d{4,5}\b', ':<PORT>', pattern)
            
            # Replace session IDs (hex patterns)
            pattern = re.sub(r'0x[0-9a-fA-F]{10,}', '<SESSION_ID>', pattern)
            
            # Replace timestamps
            pattern = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '<TIMESTAMP>', pattern)
            
            # Replace large numbers (IDs, timeouts, etc.)
            pattern = re.sub(r'\b\d{4,}\b', '<NUMBER>', pattern)
            
            # Replace file paths
            pattern = re.sub(r'/[a-zA-Z0-9_./\-]+', '<PATH>', pattern)
            
            patterns.append(pattern)
        
        df['message_pattern'] = patterns
        
        # Count pattern frequencies
        pattern_counts = Counter(patterns)
        df['pattern_frequency'] = df['message_pattern'].map(pattern_counts)
        
        # Categorize by common patterns
        df['auto_category'] = df['message_pattern'].apply(self._categorize_pattern)
        
        print(f"📊 Found {len(pattern_counts)} unique patterns")
        print(f"📋 Top patterns:")
        for pattern, count in pattern_counts.most_common(5):
            print(f"   {count:3d}x: {pattern[:80]}...")
        
        return df
    
    def _categorize_pattern(self, pattern):
        """Automatically categorize log patterns."""
        pattern_lower = pattern.lower()
        
        if 'connection' in pattern_lower and 'request' in pattern_lower:
            return 'connection_request'
        elif 'connection' in pattern_lower and 'broken' in pattern_lower:
            return 'connection_broken'
        elif 'session' in pattern_lower and ('expir' in pattern_lower or 'terminat' in pattern_lower):
            return 'session_management'
        elif 'worker' in pattern_lower and ('leaving' in pattern_lower or 'interrupt' in pattern_lower):
            return 'worker_management'
        elif 'election' in pattern_lower or 'notification' in pattern_lower:
            return 'leader_election'
        elif 'client' in pattern_lower and ('establish' in pattern_lower or 'attempt' in pattern_lower):
            return 'client_connection'
        elif 'timeout' in pattern_lower or 'time out' in pattern_lower:
            return 'timeout_events'
        elif 'processed' in pattern_lower or 'process' in pattern_lower:
            return 'request_processing'
        else:
            return 'other'


class LogEmbeddingExtractor:
    """Extract BERT embeddings optimized for log messages."""
    
    def __init__(self, model_name="distilbert-base-uncased", max_length=256):
        """
        Initialize the embedding extractor.
        
        Args:
            model_name: BERT model to use
            max_length: Maximum sequence length (increased for longer logs)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def extract_embeddings(self, texts, batch_size=8):
        """Extract BERT embeddings with optimized batch processing."""
        embeddings = []
        
        print(f"🤖 Extracting BERT embeddings for {len(texts)} log messages...")
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                try:
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
                    
                    # Use [CLS] token embedding as sentence representation
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]
                    embeddings.append(cls_embeddings.cpu().numpy())
                    
                except Exception as e:
                    print(f"⚠️  Error processing batch {i//batch_size}: {e}")
                    # Add zero embeddings for failed batch
                    embeddings.append(np.zeros((len(batch_texts), 768)))
        
        result = np.vstack(embeddings)
        print(f"✅ Extracted embeddings shape: {result.shape}")
        return result


def perform_real_log_clustering(embeddings, log_data, n_clusters_range=(3, 12)):
    """
    Perform clustering analysis on real log data.
    
    Args:
        embeddings: BERT embeddings
        log_data: DataFrame with log information
        n_clusters_range: Range of cluster numbers to test
        
    Returns:
        Dictionary with clustering results
    """
    
    print(f"\n🔍 CLUSTERING ANALYSIS ON REAL LOGS")
    print("=" * 40)
    
    # Find optimal number of clusters
    silhouette_scores = []
    cluster_range = range(n_clusters_range[0], n_clusters_range[1] + 1)
    
    print("Finding optimal number of clusters...")
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        sil_score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(sil_score)
        
        print(f"  k={n_clusters}: Silhouette={sil_score:.3f}")
    
    # Find optimal k
    optimal_k = cluster_range[np.argmax(silhouette_scores)]
    
    print(f"\n🎯 Optimal number of clusters: {optimal_k}")
    print(f"   Best silhouette score: {max(silhouette_scores):.3f}")
    
    # Perform final clustering
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(embeddings)
    
    # Add cluster labels to log data
    log_data['cluster'] = final_labels
    
    results = {
        'optimal_k': optimal_k,
        'cluster_labels': final_labels,
        'silhouette_scores': silhouette_scores,
        'cluster_range': list(cluster_range),
        'kmeans_model': final_kmeans,
        'log_data_with_clusters': log_data
    }
    
    return results


def create_real_log_visualizations(embeddings, clustering_results, log_data):
    """Create comprehensive visualizations for real log clustering."""
    
    print(f"\n📊 CREATING REAL LOG VISUALIZATIONS")
    print("=" * 40)
    
    cluster_labels = clustering_results['cluster_labels']
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("tab10")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Silhouette score plot
    ax1 = plt.subplot(3, 4, 1)
    plt.plot(clustering_results['cluster_range'], clustering_results['silhouette_scores'], 'bo-', linewidth=2)
    plt.axvline(x=clustering_results['optimal_k'], color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title(f'Optimal Clusters: k={clustering_results["optimal_k"]}')
    plt.grid(True, alpha=0.3)
    
    # 2. Log level distribution by cluster
    ax2 = plt.subplot(3, 4, 2)
    log_level_cluster = pd.crosstab(log_data['cluster'], log_data['log_level'])
    log_level_cluster.plot(kind='bar', stacked=True, ax=ax2)
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Log Levels by Cluster')
    plt.legend(title='Log Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # 3. Auto-category distribution by cluster
    ax3 = plt.subplot(3, 4, 3)
    category_cluster = pd.crosstab(log_data['cluster'], log_data['auto_category'])
    
    # Show as heatmap
    sns.heatmap(category_cluster, annot=True, fmt='d', cmap='Blues', ax=ax3)
    plt.title('Auto-Categories by Cluster')
    plt.xlabel('Auto Category')
    plt.ylabel('Cluster')
    
    # 4. PCA visualization colored by clusters
    ax4 = plt.subplot(3, 4, 5)
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Clusters (PCA)')
    plt.colorbar(scatter, label='Cluster')
    
    # 5. PCA colored by auto-category
    ax5 = plt.subplot(3, 4, 6)
    unique_categories = log_data['auto_category'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
    
    for i, category in enumerate(unique_categories):
        mask = log_data['auto_category'] == category
        plt.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                   c=[colors[i]], label=category, alpha=0.6, s=20)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Auto-Categories (PCA)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 6. t-SNE visualization
    ax6 = plt.subplot(3, 4, 9)
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Clusters (t-SNE)')
    plt.colorbar(scatter, label='Cluster')
    
    # 7. t-SNE colored by log level
    ax7 = plt.subplot(3, 4, 10)
    log_levels = log_data['log_level'].unique()
    level_colors = plt.cm.viridis(np.linspace(0, 1, len(log_levels)))
    
    for i, level in enumerate(log_levels):
        mask = log_data['log_level'] == level
        plt.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
                   c=[level_colors[i]], label=level, alpha=0.6, s=20)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Log Levels (t-SNE)')
    plt.legend()
    
    # 8. Cluster size distribution
    ax8 = plt.subplot(3, 4, 4)
    cluster_counts = log_data['cluster'].value_counts().sort_index()
    plt.bar(cluster_counts.index, cluster_counts.values, alpha=0.7)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Log Messages')
    plt.title('Cluster Sizes')
    plt.grid(True, alpha=0.3)
    
    # 9. Sample messages from each cluster
    ax9 = plt.subplot(3, 4, (7, 8))
    ax9.axis('off')
    
    sample_text = "Sample Log Messages by Cluster:\n\n"
    for cluster_id in range(clustering_results['optimal_k']):
        cluster_data = log_data[log_data['cluster'] == cluster_id]
        
        if len(cluster_data) > 0:
            sample_text += f"Cluster {cluster_id} ({len(cluster_data)} logs):\n"
            sample_text += f"  Most common category: {cluster_data['auto_category'].mode().iloc[0]}\n"
            
            # Show 2 sample messages
            for i, (_, row) in enumerate(cluster_data.head(2).iterrows()):
                msg = row['message'][:70] + "..." if len(row['message']) > 70 else row['message']
                sample_text += f"  • {row['log_level']}: {msg}\n"
            sample_text += "\n"
    
    ax9.text(0.02, 0.98, sample_text, transform=ax9.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    # 10. Pattern frequency analysis
    ax10 = plt.subplot(3, 4, (11, 12))
    top_patterns = log_data['message_pattern'].value_counts().head(10)
    
    y_pos = np.arange(len(top_patterns))
    plt.barh(y_pos, top_patterns.values, alpha=0.7)
    plt.yticks(y_pos, [p[:50] + "..." if len(p) > 50 else p for p in top_patterns.index])
    plt.xlabel('Frequency')
    plt.title('Top 10 Log Patterns')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_log_clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return embeddings_pca, embeddings_tsne


def main():
    """Main analysis function for real logs."""
    
    print("🔍 REAL ZOOKEEPER LOG CLUSTERING ANALYSIS")
    print("=" * 60)
    
    # Parse the Zookeeper log file
    parser = ZookeeperLogParser()
    log_data = parser.parse_log_file('Zookeeper_2k.log', sample_size=1000)  # Sample for speed
    
    print(f"\n📊 LOG DATA OVERVIEW:")
    print(f"   Total entries: {len(log_data)}")
    print(f"   Log levels: {log_data['log_level'].value_counts().to_dict()}")
    print(f"   Auto-categories: {log_data['auto_category'].value_counts().to_dict()}")
    print(f"   Unique patterns: {log_data['message_pattern'].nunique()}")
    
    # Initialize BERT pipeline for preprocessing
    print(f"\n🔧 INITIALIZING BERT PREPROCESSING")
    print("-" * 40)
    
    config = BERTConfig(
        model_name="distilbert-base-uncased",
        max_length=256,  # Longer for complex log messages
        batch_size=8     # Smaller batch for memory efficiency
    )
    
    pipeline = LogBERTPipeline(config)
    
    # Use the cleaned message content for embedding
    log_messages = log_data['message'].tolist()
    
    # Fit and preprocess
    pipeline.fit(log_messages)
    print("Preprocessing log messages...")
    preprocessed_result = pipeline.transform(log_messages)
    
    # Extract BERT embeddings
    print(f"\n🤖 EXTRACTING BERT EMBEDDINGS")
    print("-" * 40)
    
    embedder = LogEmbeddingExtractor(
        model_name="distilbert-base-uncased",
        max_length=256
    )
    
    # Use original messages for embedding (BERT handles preprocessing internally)
    embeddings = embedder.extract_embeddings(log_messages, batch_size=8)
    
    print(f"✅ Extracted embeddings shape: {embeddings.shape}")
    
    # Perform clustering analysis
    clustering_results = perform_real_log_clustering(
        embeddings, 
        log_data.copy(),
        n_clusters_range=(3, 10)
    )
    
    # Create visualizations
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("-" * 40)
    
    pca_coords, tsne_coords = create_real_log_visualizations(
        embeddings,
        clustering_results,
        clustering_results['log_data_with_clusters']
    )
    
    # Print summary
    print(f"\n✅ REAL LOG ANALYSIS COMPLETE!")
    print("=" * 40)
    print(f"📁 Visualization saved as 'real_log_clustering_analysis.png'")
    print(f"🎯 Found {clustering_results['optimal_k']} optimal clusters")
    print(f"📊 Silhouette score: {max(clustering_results['silhouette_scores']):.3f}")
    
    # Show cluster analysis
    log_with_clusters = clustering_results['log_data_with_clusters']
    print(f"\n📋 CLUSTER ANALYSIS:")
    for cluster_id in range(clustering_results['optimal_k']):
        cluster_data = log_with_clusters[log_with_clusters['cluster'] == cluster_id]
        top_category = cluster_data['auto_category'].mode().iloc[0]
        top_level = cluster_data['log_level'].mode().iloc[0]
        print(f"   Cluster {cluster_id}: {len(cluster_data):3d} logs, "
              f"mainly '{top_category}' ({top_level})")


if __name__ == "__main__":
    main() 