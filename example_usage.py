#!/usr/bin/env python3
"""
Simple example of log parsing and BERT vectorization
"""

from bert_vectorization_pipeline import BERTLogVectorizer, process_logs_with_vectors
import numpy as np
import pandas as pd

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

def main():
    """Demonstrate the clean BERT log vectorization pipeline."""
    
    print("Processing logs with BERT vectors...")
    
    # Option 1: One-liner for complete processing
    df = process_logs_with_vectors('logs.txt')
    
    print(f"✓ Processed {len(df)} log entries")
    print(f"✓ Vector dimension: {len(df['bert_vector'].iloc[0])}")
    
    # Show sample results
    print("\nProcessed logs:")
    display_cols = ['timestamp', 'log_level', 'message', 'cluster_id']
    print(df[display_cols].to_string(index=True))
    
    # Create similarity matrix table
    print("\n" + "="*80)
    print("SIMILARITY MATRIX TABLE (all pairwise similarities)")
    print("="*80)
    
    similarity_table = create_similarity_table(df)
    print(similarity_table.round(3).to_string())
    
    similarity_df = create_similarity_matrix(df)

    # Find most and least similar pairs
    most_similar = similarity_df.loc[similarity_df['similarity'].idxmax()]
    least_similar = similarity_df.loc[similarity_df['similarity'].idxmin()]
    
    print(f"\n🔥 Most similar pair: Msg_{most_similar['msg_1']} ↔ Msg_{most_similar['msg_2']} ({most_similar['similarity']:.3f})")
    print(f"❄️  Least similar pair: Msg_{least_similar['msg_1']} ↔ Msg_{least_similar['msg_2']} ({least_similar['similarity']:.3f})")
    
    return similarity_df, similarity_table

if __name__ == "__main__":
    try:
        similarity_df, similarity_table = main()
        
        print(f"\n📊 Tables created:")
        print(f"   - Pairwise similarities: {similarity_df.shape}")
        print(f"   - Similarity matrix: {similarity_table.shape}")
        
    except FileNotFoundError:
        print("Error: logs.txt not found")
        print("Create a sample log file with entries like:")
        print("2024-01-01 10:00:00 INFO User login successful")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install: pip install -r requirements.txt") 