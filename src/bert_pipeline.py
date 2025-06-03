"""
BERT-Ready Log Processing Pipeline
=================================

This module provides a complete pipeline for processing log datasets
and preparing them for BERT embedding and downstream ML tasks.
Includes batching, proper tokenization, and scikit-learn integration.
"""

import pandas as pd
import numpy as np
import re
import warnings
from typing import List, Dict, Tuple, Optional, Union, Iterator, Any
from pathlib import Path
import json
import pickle
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm.auto import tqdm

from .log_tokenizer import LogTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BERTConfig:
    """Configuration for BERT processing."""
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"
    batch_size: int = 32
    use_special_tokens: bool = True


class LogDataset(Dataset):
    """
    PyTorch Dataset for log entries with BERT tokenization.
    """
    
    def __init__(self, 
                 texts: List[str], 
                 labels: Optional[List[int]] = None,
                 tokenizer: Any = None,
                 max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            texts: List of preprocessed log messages
            labels: Optional list of labels for supervised learning
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize with BERT tokenizer
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


class LogPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor for log messages.
    Integrates with our existing LogTokenizer.
    """
    
    def __init__(self, 
                 preserve_special_tokens: bool = True,
                 use_lemmatization: bool = True,
                 remove_stopwords: bool = True,
                 min_token_length: int = 2,
                 join_tokens: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            preserve_special_tokens: Whether to preserve special tokens
            use_lemmatization: Whether to use lemmatization
            remove_stopwords: Whether to remove stopwords
            min_token_length: Minimum token length
            join_tokens: Whether to join tokens back into string
        """
        self.preserve_special_tokens = preserve_special_tokens
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.join_tokens = join_tokens
        self.tokenizer_ = None
        
    def fit(self, X, y=None):
        """Fit the preprocessor (initialize tokenizer)."""
        self.tokenizer_ = LogTokenizer(
            preserve_special_tokens=self.preserve_special_tokens,
            use_lemmatization=self.use_lemmatization,
            remove_stopwords=self.remove_stopwords,
            min_token_length=self.min_token_length
        )
        return self
    
    def transform(self, X):
        """Transform log messages."""
        if self.tokenizer_ is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()
        
        processed_texts = []
        for text in tqdm(X, desc="Preprocessing logs"):
            result = self.tokenizer_.tokenize_log_message(str(text))
            if self.join_tokens:
                processed_texts.append(result['tokenized_string'])
            else:
                processed_texts.append(result['tokens'])
        
        return processed_texts


class BERTTokenizer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible BERT tokenizer.
    """
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 max_length: int = 512,
                 padding: str = "max_length",
                 truncation: bool = True):
        """
        Initialize BERT tokenizer.
        
        Args:
            model_name: BERT model name from Hugging Face
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
        """
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.tokenizer_ = None
        
    def fit(self, X, y=None):
        """Initialize BERT tokenizer."""
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        return self
    
    def transform(self, X):
        """Tokenize texts for BERT."""
        if self.tokenizer_ is None:
            raise ValueError("BERT tokenizer must be fitted before transform")
        
        if isinstance(X, (pd.Series, np.ndarray)):
            X = X.tolist()
        
        # Batch tokenization for efficiency
        encodings = self.tokenizer_(
            X,
            add_special_tokens=True,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }


class LogBERTPipeline:
    """
    Complete pipeline for processing logs and preparing them for BERT.
    """
    
    def __init__(self, config: Optional[BERTConfig] = None):
        """
        Initialize pipeline.
        
        Args:
            config: BERT configuration
        """
        self.config = config or BERTConfig()
        self.preprocessing_pipeline = None
        self.bert_tokenizer = None
        self.statistics_ = {}
        
        # Initialize components
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup the preprocessing pipeline."""
        self.preprocessing_pipeline = Pipeline([
            ('log_preprocessor', LogPreprocessor(
                preserve_special_tokens=True,
                use_lemmatization=True,
                remove_stopwords=True,
                min_token_length=2,
                join_tokens=True
            )),
            ('bert_tokenizer', BERTTokenizer(
                model_name=self.config.model_name,
                max_length=self.config.max_length,
                padding=self.config.padding,
                truncation=self.config.truncation
            ))
        ])
        
        # Initialize BERT tokenizer for dataset creation
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
    
    def fit(self, X: Union[pd.DataFrame, List[str], np.ndarray], y=None):
        """
        Fit the pipeline.
        
        Args:
            X: Input data (DataFrame, list, or array)
            y: Optional target values
        """
        logger.info("Fitting log processing pipeline...")
        
        # Extract text column if DataFrame
        if isinstance(X, pd.DataFrame):
            if 'log_message' in X.columns:
                texts = X['log_message'].tolist()
            elif 'message' in X.columns:
                texts = X['message'].tolist()
            else:
                # Assume first column is text
                texts = X.iloc[:, 0].tolist()
        else:
            texts = X
        
        # Fit preprocessing pipeline
        self.preprocessing_pipeline.fit(texts)
        
        # Calculate statistics
        self.statistics_ = self._calculate_statistics(texts)
        
        logger.info(f"Pipeline fitted on {len(texts)} samples")
        return self
    
    def transform(self, X: Union[pd.DataFrame, List[str], np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Transform data through the pipeline.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        # Extract text column if DataFrame
        if isinstance(X, pd.DataFrame):
            if 'log_message' in X.columns:
                texts = X['log_message'].tolist()
            elif 'message' in X.columns:
                texts = X['message'].tolist()
            else:
                texts = X.iloc[:, 0].tolist()
        else:
            texts = X
        
        # Transform through pipeline
        return self.preprocessing_pipeline.transform(texts)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def create_dataset(self, 
                      X: Union[pd.DataFrame, List[str], np.ndarray],
                      y: Optional[Union[List, np.ndarray]] = None,
                      transform: bool = True) -> LogDataset:
        """
        Create PyTorch dataset.
        
        Args:
            X: Input data
            y: Optional labels
            transform: Whether to apply preprocessing
            
        Returns:
            LogDataset ready for DataLoader
        """
        if transform:
            # Apply log preprocessing but not BERT tokenization
            # (BERT tokenization happens in dataset __getitem__)
            log_preprocessor = LogPreprocessor(
                preserve_special_tokens=True,
                use_lemmatization=True,
                remove_stopwords=True,
                min_token_length=2,
                join_tokens=True
            )
            
            # Extract texts
            if isinstance(X, pd.DataFrame):
                if 'log_message' in X.columns:
                    texts = X['log_message'].tolist()
                elif 'message' in X.columns:
                    texts = X['message'].tolist()
                else:
                    texts = X.iloc[:, 0].tolist()
            else:
                texts = X
            
            # Fit and transform
            log_preprocessor.fit(texts)
            processed_texts = log_preprocessor.transform(texts)
        else:
            processed_texts = X
        
        return LogDataset(
            texts=processed_texts,
            labels=y,
            tokenizer=self.bert_tokenizer,
            max_length=self.config.max_length
        )
    
    def create_dataloader(self, 
                         dataset: LogDataset,
                         batch_size: Optional[int] = None,
                         shuffle: bool = True,
                         num_workers: int = 0) -> DataLoader:
        """
        Create PyTorch DataLoader.
        
        Args:
            dataset: LogDataset
            batch_size: Batch size (uses config default if None)
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader for batch processing
        """
        batch_size = batch_size or self.config.batch_size
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        batch_data = {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'texts': [item['text'] for item in batch]
        }
        
        if 'labels' in batch[0]:
            batch_data['labels'] = torch.stack([item['labels'] for item in batch])
        
        return batch_data
    
    def process_dataset_in_batches(self, 
                                  X: Union[pd.DataFrame, List[str], np.ndarray],
                                  batch_size: Optional[int] = None,
                                  save_path: Optional[str] = None) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Process large dataset in batches.
        
        Args:
            X: Input data
            batch_size: Batch size
            save_path: Optional path to save processed batches
            
        Yields:
            Batches of processed data
        """
        batch_size = batch_size or self.config.batch_size
        
        # Extract texts
        if isinstance(X, pd.DataFrame):
            if 'log_message' in X.columns:
                texts = X['log_message'].tolist()
            elif 'message' in X.columns:
                texts = X['message'].tolist()
            else:
                texts = X.iloc[:, 0].tolist()
        else:
            texts = X
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Process batch
            batch_result = self.transform(batch_texts)
            
            # Save if requested
            if save_path:
                batch_path = f"{save_path}/batch_{i//batch_size:04d}.pt"
                torch.save(batch_result, batch_path)
            
            yield batch_result
    
    def _calculate_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Calculate dataset statistics."""
        stats = {
            'total_samples': len(texts),
            'avg_text_length': np.mean([len(text) for text in texts]),
            'max_text_length': max([len(text) for text in texts]),
            'min_text_length': min([len(text) for text in texts]),
        }
        
        # Token statistics (approximate)
        sample_texts = texts[:min(1000, len(texts))]
        token_lengths = []
        
        for text in sample_texts:
            tokens = self.bert_tokenizer.tokenize(text)
            token_lengths.append(len(tokens))
        
        stats.update({
            'avg_token_length': np.mean(token_lengths),
            'max_token_length': max(token_lengths),
            'tokens_exceeding_limit': sum(1 for l in token_lengths if l > self.config.max_length)
        })
        
        return stats
    
    def save_pipeline(self, path: str):
        """Save pipeline to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline state
        pipeline_state = {
            'config': self.config,
            'statistics': self.statistics_,
            'preprocessing_pipeline': self.preprocessing_pipeline
        }
        
        with open(path, 'wb') as f:
            pickle.dump(pipeline_state, f)
        
        logger.info(f"Pipeline saved to {path}")
    
    @classmethod
    def load_pipeline(cls, path: str):
        """Load pipeline from disk."""
        with open(path, 'rb') as f:
            pipeline_state = pickle.load(f)
        
        # Recreate pipeline
        pipeline = cls(config=pipeline_state['config'])
        pipeline.preprocessing_pipeline = pipeline_state['preprocessing_pipeline']
        pipeline.statistics_ = pipeline_state['statistics']
        
        logger.info(f"Pipeline loaded from {path}")
        return pipeline


class LogClassificationPipeline(LogBERTPipeline):
    """
    Extended pipeline for log classification tasks.
    """
    
    def __init__(self, 
                 config: Optional[BERTConfig] = None,
                 num_classes: Optional[int] = None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize classification pipeline.
        
        Args:
            config: BERT configuration
            num_classes: Number of classes
            class_names: List of class names
        """
        super().__init__(config)
        self.num_classes = num_classes
        self.class_names = class_names
        self.label_encoder_ = None
    
    def fit(self, X, y):
        """Fit pipeline with labels."""
        # Fit base pipeline
        super().fit(X)
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder_ = LabelEncoder()
        encoded_labels = self.label_encoder_.fit_transform(y)
        
        # Update class information
        self.num_classes = len(self.label_encoder_.classes_)
        self.class_names = self.label_encoder_.classes_.tolist()
        
        logger.info(f"Classification pipeline fitted with {self.num_classes} classes")
        return self
    
    def prepare_train_val_split(self, 
                               X, y, 
                               test_size: float = 0.2,
                               random_state: int = 42,
                               stratify: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare train/validation split with DataLoaders.
        
        Args:
            X: Input data
            y: Labels
            test_size: Fraction for validation
            random_state: Random seed
            stratify: Whether to stratify split
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Encode labels
        encoded_labels = self.label_encoder_.transform(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, encoded_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=encoded_labels if stratify else None
        )
        
        # Create datasets
        train_dataset = self.create_dataset(X_train, y_train, transform=True)
        val_dataset = self.create_dataset(X_val, y_val, transform=True)
        
        # Create data loaders
        train_loader = self.create_dataloader(train_dataset, shuffle=True)
        val_loader = self.create_dataloader(val_dataset, shuffle=False)
        
        return train_loader, val_loader


def demonstrate_bert_pipeline():
    """
    Demonstrate the BERT pipeline with sample data.
    """
    print("🤖 BERT LOG PROCESSING PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Sample log data
    sample_logs = [
        "2024-12-24 14:30:22 INFO User john.doe@company.com successfully authenticated from IP 10.0.1.15:443",
        "2024-12-24 14:30:25 ERROR Database connection failed: timeout after 30s connecting to db.internal:5432",
        "2024-12-24 14:30:28 DEBUG GET /api/v2/users/12345/profile.json completed in 245ms",
        "2024-12-24 14:30:31 CRITICAL Suspicious activity detected: 15 failed login attempts from 192.168.100.50",
        "2024-12-24 14:30:34 WARN High memory usage: JVM heap at 85% (3.2GB/3.8GB), PID 8472",
        "2024-12-24 14:30:37 INFO Successfully archived 147 files (512MB) to /backup/daily/2024-12-24.tar.gz",
    ] * 10  # Multiply for demonstration
    
    # Sample labels (log levels)
    sample_labels = ['INFO', 'ERROR', 'DEBUG', 'CRITICAL', 'WARN', 'INFO'] * 10
    
    # Create DataFrame
    df = pd.DataFrame({
        'log_message': sample_logs,
        'log_level': sample_labels
    })
    
    print(f"📊 Sample dataset: {len(df)} log entries")
    print(f"📋 Columns: {df.columns.tolist()}")
    print(f"🏷️  Labels: {df['log_level'].unique()}")
    
    # Initialize pipeline
    config = BERTConfig(
        model_name="distilbert-base-uncased",  # Smaller model for demo
        max_length=256,
        batch_size=8
    )
    
    print(f"\n🔧 Pipeline Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Max length: {config.max_length}")
    print(f"   Batch size: {config.batch_size}")
    
    # Basic pipeline demonstration
    print(f"\n🔹 BASIC PIPELINE DEMO")
    print("-" * 30)
    
    pipeline = LogBERTPipeline(config)
    pipeline.fit(df['log_message'])
    
    # Transform sample
    sample_batch = df['log_message'].head(3).tolist()
    result = pipeline.transform(sample_batch)
    
    print(f"Input texts: {len(sample_batch)}")
    print(f"Output shape: {result['input_ids'].shape}")
    print(f"Attention mask shape: {result['attention_mask'].shape}")
    
    # Dataset creation
    print(f"\n🔹 DATASET CREATION DEMO")
    print("-" * 30)
    
    dataset = pipeline.create_dataset(
        df['log_message'].head(5),
        transform=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Show sample
    sample_item = dataset[0]
    print(f"Sample item keys: {sample_item.keys()}")
    print(f"Input IDs shape: {sample_item['input_ids'].shape}")
    print(f"Sample text: {sample_item['text'][:100]}...")
    
    # DataLoader demo
    print(f"\n🔹 DATALOADER DEMO")
    print("-" * 30)
    
    dataloader = pipeline.create_dataloader(dataset, batch_size=2, shuffle=False)
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Input IDs: {batch['input_ids'].shape}")
        print(f"  Attention mask: {batch['attention_mask'].shape}")
        print(f"  Texts: {len(batch['texts'])}")
        if i >= 1:  # Show only first 2 batches
            break
    
    # Classification pipeline demo
    print(f"\n🔹 CLASSIFICATION PIPELINE DEMO")
    print("-" * 30)
    
    clf_pipeline = LogClassificationPipeline(config)
    clf_pipeline.fit(df['log_message'], df['log_level'])
    
    print(f"Number of classes: {clf_pipeline.num_classes}")
    print(f"Class names: {clf_pipeline.class_names}")
    
    # Train/val split
    train_loader, val_loader = clf_pipeline.prepare_train_val_split(
        df['log_message'], df['log_level'], 
        test_size=0.3
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Show statistics
    print(f"\n📈 PIPELINE STATISTICS")
    print("-" * 30)
    stats = pipeline.statistics_
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n✅ Pipeline demonstration complete!")


if __name__ == "__main__":
    demonstrate_bert_pipeline() 