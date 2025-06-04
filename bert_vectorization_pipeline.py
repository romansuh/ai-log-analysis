#!/usr/bin/env python3
"""
Clean BERT Vectorization Pipeline for Log Messages
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Optional

class BERTLogVectorizer:
    """Simple BERT vectorizer for log messages."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """Initialize with BERT model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def vectorize(self, message: str) -> np.ndarray:
        """Convert a log message to BERT vector."""
        if not message or pd.isna(message):
            return np.zeros(self.model.config.hidden_size)
        
        inputs = self.tokenizer(message, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    def add_to_dataframe(self, df: pd.DataFrame, column: str = 'message') -> pd.DataFrame:
        """Add BERT vectors to dataframe."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        vectors = [self.vectorize(msg) for msg in df[column].fillna('')]
        df = df.copy()
        df['bert_vector'] = vectors
        return df

def process_logs_with_vectors(file_path: str) -> pd.DataFrame:
    """Complete pipeline: parse logs and add BERT vectors."""
    from log_parsing_pipeline import LogParsingPipeline
    
    # Parse logs
    parser = LogParsingPipeline()
    df = parser.process_logs(file_path)
    
    # Add BERT vectors
    vectorizer = BERTLogVectorizer()
    return vectorizer.add_to_dataframe(df)

if __name__ == "__main__":
    # Simple usage
    df = process_logs_with_vectors('logs.txt')
    
    print(f"Processed {len(df)} log entries")
    print(f"Vector dimension: {len(df['bert_vector'].iloc[0]) if len(df) > 0 else 'N/A'}")
    
    # Show results (without vectors for readability)
    columns = [col for col in df.columns if col != 'bert_vector']
    print("\nProcessed logs:")
    print(df[columns].head().to_string(index=False)) 