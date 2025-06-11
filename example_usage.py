#!/usr/bin/env python3
"""
Simple example of log parsing and BERT vectorization
"""

from bert_vectorization_pipeline import BERTLogVectorizer, process_logs_with_vectors
import numpy as np
import pandas as pd
from cluster_visualization import LogClusterVisualizer, add_kmeans_clusters
import matplotlib.pyplot as plt

N_CLUSTERS = 44

def create_similarity_matrix(df):
    """Create a dataframe with similarities between all pairs of log messages."""
    vectors = np.vstack(df['bert_vector'].tolist())
    n = len(vectors)
    
    # Calculate cosine similarities
    similarities = []
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip self-similarity
                similarity = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                similarities.append({
                    'msg_1': i,
                    'msg_2': j,
                    'similarity': similarity
                })
    
    return pd.DataFrame(similarities)

def create_similarity_table(df):
    """Create a square matrix table showing all pairwise similarities."""
    vectors = np.vstack(df['bert_vector'].tolist())
    n = len(vectors)
    
    # Create similarity matrix
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i][j] = 1.0  # Self-similarity
            else:
                similarity = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                similarity_matrix[i][j] = similarity
    
    # Create DataFrame with message indices as row and column names
    return pd.DataFrame(
        similarity_matrix, 
        index=[f'Msg_{i}' for i in range(n)],
        columns=[f'Msg_{i}' for i in range(n)]
    )

def plot_clusters(embeddings, labels, title):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.6)

    for i in range(0, len(reduced), 200):
        plt.annotate(f'Msg_{i}', (reduced[i, 0], reduced[i, 1]))

    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()

def main():
    """Demonstrate the clean BERT log vectorization pipeline."""
    
    print("Processing logs with BERT vectors...")
    
    # Option 1: One-liner for complete processing
    df = process_logs_with_vectors('Zookeeper_2k.log')
    
    print(f"✓ Processed {len(df)} log entries")
    print(f"✓ Vector dimension: {len(df['bert_vector'].iloc[0])}")
    
    # Show sample results
    # print("\nProcessed logs (first 5 entries):")
    # display_cols = ['timestamp', 'log_level', 'message', 'Drain_cluster_id', 'template']
    # print(df[display_cols].head(5).to_string(index=True))
    
    # Add KMeans clusters and visualize
    print("\nVisualizing clusters (KMeans on BERT embeddings)...")
    df = add_kmeans_clusters(df, n_clusters=N_CLUSTERS)
    
    # Visualize KMeans clusters
    # embeddings = np.vstack(df['bert_vector'].values)
    # plot_clusters(embeddings, np.vstack(df['KMeans_cluster_id'].values), title='KMeans Clusters on BERT embeddings')
    
    # Analyze KMeans clusters
    print("KMeans Cluster Sizes:")
    kmeans_cluster_sizes = pd.Series(df['KMeans_cluster_id']).value_counts().sort_index()
    for cluster, size in kmeans_cluster_sizes.items():
        print(f"Cluster {cluster}: {size} messages ({size/len(df)*100:.1f}%)")
    
    # Visualize clusters (Drain cluster_id)
    # print("\nVisualizing clusters (Drain cluster_id on BERT embeddings)...")
    # cluster_ids = df['Drain_cluster_id'].astype('category').cat.codes.values
    # plot_clusters(np.vstack(df['bert_vector'].values), cluster_ids, title='Clusterization with Drain cluster_id')
    
    # Analyze Drain clusters
    # print("\nDrain Cluster Sizes:")
    # drain_cluster_sizes = df['Drain_cluster_id'].value_counts().sort_index()
    # for cluster, size in drain_cluster_sizes.items():
    #     print(f"Cluster {cluster}: {size} messages ({size/len(df)*100:.1f}%)")
    
    # Create similarity matrix table
    # print("\n" + "="*80)
    # print("SIMILARITY MATRIX TABLE (all pairwise similarities)")
    # print("="*80)
    
    # similarity_table = create_similarity_table(df)
    # print(similarity_table.head(10).round(3).to_string())
    
    # similarity_df = create_similarity_matrix(df)
    
    # Save processed data to CSV with both cluster IDs
    output_file = 'processed_logs.csv'
    display_cols = ['log_level', 'message', 'Drain_cluster_id', 'KMeans_cluster_id', 'template']
    df[display_cols].to_csv(output_file, index=False)
    print(f"\nProcessed data has been saved to {output_file}")
    
    # Save similarity matrix to CSV
    # similarity_file = 'similarity_matrix.csv'
    # similarity_df.to_csv(similarity_file)
    # print(f"Similarity matrix has been saved to {similarity_file}")
    
    # Find most and least similar pairs
    # most_similar = similarity_df.loc[similarity_df['similarity'].idxmax()]
    # least_similar = similarity_df.loc[similarity_df['similarity'].idxmin()]
    
    # print(f"\n🔥 Most similar pair: Msg_{most_similar['msg_1']} ↔ Msg_{most_similar['msg_2']} ({most_similar['similarity']:.3f})")
    # print(f"❄️  Least similar pair: Msg_{least_similar['msg_1']} ↔ Msg_{least_similar['msg_2']} ({least_similar['similarity']:.3f})")
    
    # return similarity_df, similarity_table

if __name__ == "__main__":
    try:
        main()
        # similarity_df, similarity_table = main()
        
        # print(f"\n📊 Tables created:")
        # print(f"   - Pairwise similarities: {similarity_df.shape}")
        # print(f"   - Similarity matrix: {similarity_table.shape}")
        
    except FileNotFoundError:
        print("Error: logs.txt not found")
        print("Create a sample log file with entries like:")
        print("2024-01-01 10:00:00 INFO User login successful")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install: pip install -r requirements.txt") 