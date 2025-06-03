#!/usr/bin/env python3
"""
BERT Pipeline Demonstration
===========================

This script demonstrates the complete BERT-ready log processing pipeline
including data input/output formats, batching, and ML integration.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import time
from typing import List, Dict

# Import our BERT pipeline (note: we'll need to install transformers first)
try:
    from src.bert_pipeline import (
        BERTConfig, 
        LogBERTPipeline, 
        LogClassificationPipeline,
        LogDataset
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Transformers not available: {e}")
    print("📦 Install with: pip install transformers torch tqdm")
    TRANSFORMERS_AVAILABLE = False

from src.log_tokenizer import LogTokenizer


def generate_sample_dataset(size: int = 1000) -> pd.DataFrame:
    """
    Generate a realistic sample log dataset for demonstration.
    
    Args:
        size: Number of log entries to generate
        
    Returns:
        DataFrame with log messages and labels
    """
    
    print(f"📊 Generating sample dataset with {size} entries...")
    
    # Base log templates
    log_templates = {
        'INFO': [
            "User {user}@{domain} successfully authenticated from IP {ip}:{port}",
            "Successfully archived {count} files ({size}) to {path}",
            "Backup job completed in {duration}, total size: {size}",
            "User {user} accessed {endpoint} successfully",
            "Service {service} started on port {port}",
        ],
        'ERROR': [
            "Database connection failed: timeout after {duration} connecting to {host}:{port}",
            "Failed to process request to {endpoint}: {error_code}",
            "Authentication failed for user {user} from IP {ip}",
            "Unable to write to {path}: permission denied",
            "Service {service} crashed with exit code {code}",
        ],
        'WARNING': [
            "High memory usage: {service} heap at {percentage}% ({current}/{total}), PID {pid}",
            "SSL certificate for {domain} expires in {days} days",
            "Disk usage at {percentage}% on {path}",
            "Slow query detected: {duration} for {query}",
            "Rate limit exceeded for IP {ip}: {count} requests",
        ],
        'DEBUG': [
            "GET {endpoint} completed in {duration}, response size: {size}",
            "Processing {count} items in batch {batch_id}",
            "Cache hit for key {key}, response time: {duration}",
            "Executing query: {query} on database {db}",
            "Loading configuration from {path}",
        ],
        'CRITICAL': [
            "Suspicious activity detected: {count} failed login attempts from {ip}",
            "Memory allocation failed at {address}, PID {pid} killed by OOM",
            "Security breach detected: unauthorized access to {resource}",
            "Data corruption detected in {path}",
            "System overload: {count} concurrent connections exceeded limit",
        ]
    }
    
    # Sample data for templates
    sample_data = {
        'user': ['admin', 'john.doe', 'alice.smith', 'bob.jones', 'system', 'api_user'],
        'domain': ['company.com', 'internal.local', 'api.service.com', 'localhost'],
        'ip': ['192.168.1.100', '10.0.0.50', '172.16.0.1', '203.0.113.1', '198.51.100.1'],
        'port': ['443', '8080', '5432', '3306', '80', '22', '9200'],
        'path': ['/var/log/app.log', '/backup/daily/2024-12-24.tar.gz', '/data/users.db', '/config/app.conf'],
        'endpoint': ['/api/v1/users', '/api/v2/data', '/health', '/login', '/logout'],
        'service': ['nginx', 'postgres', 'redis', 'elasticsearch', 'api-gateway'],
        'duration': ['30s', '245ms', '1.2hr', '5min', '500ms', '2.5s'],
        'size': ['2.1KB', '512MB', '1.2GB', '15MB', '64KB'],
        'error_code': ['E5001', 'DB_CONN_ERR_001', 'AUTH_FAILED', 'TIMEOUT_ERROR'],
        'percentage': ['85', '92', '78', '95', '67'],
        'count': ['15', '147', '1024', '50', '3'],
        'pid': ['8472', '12345', '9876', '5432'],
        'days': ['7', '15', '30', '2'],
        'code': ['137', '1', '255', '0'],
        'batch_id': ['batch_001', 'batch_002', 'batch_003'],
        'key': ['user_123', 'session_456', 'cache_789'],
        'query': ['SELECT * FROM users', 'UPDATE logs SET processed=true', 'DELETE FROM temp'],
        'db': ['userdb', 'logdb', 'analytics'],
        'address': ['0x7fff5fbff7e0', '0xDEADBEEF', '0x12345678'],
        'resource': ['/admin/panel', '/secure/data', '/config/secrets'],
        'host': ['db.internal', 'cache.local', 'api.service.com']
    }
    
    logs = []
    labels = []
    
    # Generate random logs
    for _ in range(size):
        # Random log level
        level = np.random.choice(list(log_templates.keys()), p=[0.4, 0.2, 0.2, 0.15, 0.05])
        
        # Random template
        template = np.random.choice(log_templates[level])
        
        # Fill template with random data
        filled_template = template
        for key, values in sample_data.items():
            if '{' + key + '}' in template:
                filled_template = filled_template.replace('{' + key + '}', np.random.choice(values))
        
        # Add timestamp
        timestamp = f"2024-12-24 {np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}:{np.random.randint(0, 60):02d}"
        log_entry = f"{timestamp} {level} {filled_template}"
        
        logs.append(log_entry)
        labels.append(level)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-12-24', periods=size, freq='1min'),
        'log_message': logs,
        'log_level': labels,
        'message_length': [len(msg) for msg in logs]
    })
    
    print(f"✅ Generated dataset with {len(df)} entries")
    print(f"📋 Log levels: {df['log_level'].value_counts().to_dict()}")
    print(f"📏 Avg message length: {df['message_length'].mean():.1f} characters")
    
    return df


def demonstrate_input_output_formats():
    """
    Demonstrate various input and output formats supported by the pipeline.
    """
    
    print("\n" + "=" * 60)
    print("📁 INPUT/OUTPUT FORMATS DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    df = generate_sample_dataset(100)
    
    print(f"\n🔹 INPUT FORMATS SUPPORTED:")
    print("-" * 30)
    
    # Format 1: pandas DataFrame
    print("1. 📊 Pandas DataFrame (recommended)")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Sample:")
    print(f"   {df['log_message'].iloc[0][:80]}...")
    
    # Format 2: List of strings
    text_list = df['log_message'].tolist()[:10]
    print(f"\n2. 📝 List of strings")
    print(f"   Type: {type(text_list)}")
    print(f"   Length: {len(text_list)}")
    print(f"   Sample: {text_list[0][:80]}...")
    
    # Format 3: NumPy array
    text_array = df['log_message'].values[:10]
    print(f"\n3. 🔢 NumPy array")
    print(f"   Type: {type(text_array)}")
    print(f"   Shape: {text_array.shape}")
    print(f"   Sample: {text_array[0][:80]}...")
    
    # Format 4: CSV file
    csv_path = "data/sample_logs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n4. 📄 CSV file")
    print(f"   Path: {csv_path}")
    print(f"   Size: {Path(csv_path).stat().st_size / 1024:.1f} KB")
    
    if not TRANSFORMERS_AVAILABLE:
        print("\n⚠️  Skipping BERT pipeline demo (transformers not installed)")
        return
    
    print(f"\n🔹 OUTPUT FORMATS:")
    print("-" * 30)
    
    # Initialize pipeline
    config = BERTConfig(
        model_name="distilbert-base-uncased",
        max_length=128,
        batch_size=8
    )
    
    pipeline = LogBERTPipeline(config)
    pipeline.fit(df['log_message'].head(20))
    
    # Output 1: BERT tensors
    sample_texts = df['log_message'].head(3).tolist()
    bert_output = pipeline.transform(sample_texts)
    
    print("1. 🤖 BERT Tensors")
    print(f"   Input IDs shape: {bert_output['input_ids'].shape}")
    print(f"   Attention mask shape: {bert_output['attention_mask'].shape}")
    print(f"   Data type: {bert_output['input_ids'].dtype}")
    
    # Output 2: PyTorch Dataset
    dataset = pipeline.create_dataset(df['log_message'].head(5), transform=True)
    
    print(f"\n2. 🔥 PyTorch Dataset")
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Sample item keys: {list(dataset[0].keys())}")
    print(f"   Input tensor shape: {dataset[0]['input_ids'].shape}")
    
    # Output 3: DataLoader batches
    dataloader = pipeline.create_dataloader(dataset, batch_size=2)
    
    print(f"\n3. 📦 DataLoader Batches")
    print(f"   Number of batches: {len(dataloader)}")
    
    for i, batch in enumerate(dataloader):
        print(f"   Batch {i+1}: {batch['input_ids'].shape}")
        if i >= 1:
            break
    
    # Output 4: Saved batches
    batch_dir = Path("results/bert_batches")
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n4. 💾 Saved Batch Files")
    batch_count = 0
    for batch in pipeline.process_dataset_in_batches(
        df['log_message'].head(10), 
        batch_size=5, 
        save_path=str(batch_dir)
    ):
        batch_count += 1
    
    saved_files = list(batch_dir.glob("*.pt"))
    print(f"   Saved {len(saved_files)} batch files to {batch_dir}")
    
    if saved_files:
        sample_batch = torch.load(saved_files[0])
        print(f"   Sample batch shape: {sample_batch['input_ids'].shape}")


def demonstrate_batching_strategies():
    """
    Demonstrate different batching strategies for large datasets.
    """
    
    if not TRANSFORMERS_AVAILABLE:
        print("\n⚠️  Skipping batching demo (transformers not installed)")
        return
    
    print("\n" + "=" * 60)
    print("📦 BATCHING STRATEGIES DEMONSTRATION")
    print("=" * 60)
    
    # Generate larger dataset
    df = generate_sample_dataset(500)
    
    config = BERTConfig(
        model_name="distilbert-base-uncased",
        max_length=128,
        batch_size=16
    )
    
    pipeline = LogBERTPipeline(config)
    pipeline.fit(df['log_message'].head(50))
    
    print(f"\n🔹 STRATEGY 1: MEMORY-EFFICIENT STREAMING")
    print("-" * 40)
    
    start_time = time.time()
    batch_count = 0
    total_samples = 0
    
    for batch in pipeline.process_dataset_in_batches(df['log_message'], batch_size=32):
        batch_count += 1
        total_samples += batch['input_ids'].shape[0]
        
        # Simulate processing
        if batch_count <= 3:  # Show first few batches
            print(f"   Batch {batch_count}: {batch['input_ids'].shape} - Memory: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 'N/A'}")
    
    processing_time = time.time() - start_time
    print(f"   Total: {batch_count} batches, {total_samples} samples in {processing_time:.2f}s")
    
    print(f"\n🔹 STRATEGY 2: DATALOADER WITH WORKERS")
    print("-" * 40)
    
    # Create dataset and dataloader
    dataset = pipeline.create_dataset(df['log_message'], transform=True)
    dataloader = pipeline.create_dataloader(dataset, batch_size=32, num_workers=0)  # 0 for demo
    
    start_time = time.time()
    batch_count = 0
    
    for batch in dataloader:
        batch_count += 1
        if batch_count <= 3:
            print(f"   Batch {batch_count}: {batch['input_ids'].shape}")
        if batch_count >= 5:  # Limit for demo
            break
    
    processing_time = time.time() - start_time
    print(f"   Processed {batch_count} batches in {processing_time:.2f}s")
    
    print(f"\n🔹 STRATEGY 3: ADAPTIVE BATCHING")
    print("-" * 40)
    
    # Adaptive batch sizes based on text length
    df['text_length'] = df['log_message'].str.len()
    df_sorted = df.sort_values('text_length')
    
    print("   Batching by text length for efficiency:")
    
    # Short texts (larger batches)
    short_texts = df_sorted[df_sorted['text_length'] < 100]['log_message'].head(50)
    short_dataset = pipeline.create_dataset(short_texts, transform=True)
    short_loader = pipeline.create_dataloader(short_dataset, batch_size=64)
    
    print(f"   Short texts (<100 chars): {len(short_texts)} samples, batch_size=64")
    print(f"   Batches: {len(short_loader)}")
    
    # Long texts (smaller batches) 
    long_texts = df_sorted[df_sorted['text_length'] >= 200]['log_message'].head(40)
    if len(long_texts) > 0:
        long_dataset = pipeline.create_dataset(long_texts, transform=True)
        long_loader = pipeline.create_dataloader(long_dataset, batch_size=16)
        
        print(f"   Long texts (≥200 chars): {len(long_texts)} samples, batch_size=16")
        print(f"   Batches: {len(long_loader)}")


def demonstrate_ml_integration():
    """
    Demonstrate integration with ML workflows.
    """
    
    if not TRANSFORMERS_AVAILABLE:
        print("\n⚠️  Skipping ML integration demo (transformers not installed)")
        return
    
    print("\n" + "=" * 60)
    print("🤖 ML INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Generate classification dataset
    df = generate_sample_dataset(200)
    
    print(f"\n🔹 LOG CLASSIFICATION PIPELINE")
    print("-" * 40)
    
    # Initialize classification pipeline
    config = BERTConfig(
        model_name="distilbert-base-uncased",
        max_length=128,
        batch_size=16
    )
    
    clf_pipeline = LogClassificationPipeline(config)
    clf_pipeline.fit(df['log_message'], df['log_level'])
    
    print(f"Classes: {clf_pipeline.class_names}")
    print(f"Number of classes: {clf_pipeline.num_classes}")
    
    # Train/validation split
    train_loader, val_loader = clf_pipeline.prepare_train_val_split(
        df['log_message'], df['log_level'],
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Show sample batch
    sample_batch = next(iter(train_loader))
    print(f"Sample batch:")
    print(f"  Input shape: {sample_batch['input_ids'].shape}")
    print(f"  Labels shape: {sample_batch['labels'].shape}")
    print(f"  Label distribution: {torch.bincount(sample_batch['labels'].flatten())}")
    
    print(f"\n🔹 INTEGRATION WITH SCIKIT-LEARN")
    print("-" * 40)
    
    # Use pipeline as preprocessing step
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    
    # Extract BERT features (this would typically be done with a trained BERT model)
    sample_texts = df['log_message'].head(100).tolist()
    sample_labels = df['log_level'].head(100).tolist()
    
    # For demo, we'll use our log tokenizer features
    from src.log_tokenizer import LogTokenizer
    tokenizer = LogTokenizer()
    
    # Create feature matrix from tokens
    features = []
    for text in sample_texts:
        result = tokenizer.tokenize_log_message(text)
        # Simple feature: token count and special token count
        features.append([
            result['token_stats']['total_tokens'],
            result['token_stats']['special_token_types'],
            len(result['original_text'])
        ])
    
    X = np.array(features)
    y = sample_labels
    
    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print("Feature-based classification results:")
    print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\n🔹 PIPELINE STATISTICS & MONITORING")
    print("-" * 40)
    
    stats = clf_pipeline.statistics_
    print("Pipeline Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def demonstrate_real_world_usage():
    """
    Demonstrate real-world usage patterns and best practices.
    """
    
    print("\n" + "=" * 60)
    print("🌍 REAL-WORLD USAGE PATTERNS")
    print("=" * 60)
    
    print(f"\n🔹 SCENARIO 1: LOG ANOMALY DETECTION")
    print("-" * 40)
    
    # Generate dataset with anomalies
    normal_logs = generate_sample_dataset(800)
    
    # Create some anomalous logs
    anomalous_logs = [
        "2024-12-24 15:30:00 ERROR Unexpected system call: execve('/bin/sh', ...) from PID 9999",
        "2024-12-24 15:31:00 CRITICAL Multiple failed sudo attempts from unknown user 'h4cker'",
        "2024-12-24 15:32:00 WARNING Large file transfer: 10GB uploaded to /tmp/suspicious.dat",
        "2024-12-24 15:33:00 ERROR SQL injection attempt detected: SELECT * FROM users WHERE '1'='1'",
    ]
    
    # Combine datasets
    anomaly_df = pd.DataFrame({
        'log_message': normal_logs['log_message'].tolist() + anomalous_logs,
        'is_anomaly': [0] * len(normal_logs) + [1] * len(anomalous_logs)
    })
    
    print(f"Dataset: {len(anomaly_df)} logs ({anomaly_df['is_anomaly'].sum()} anomalies)")
    
    if TRANSFORMERS_AVAILABLE:
        # Setup pipeline for anomaly detection
        config = BERTConfig(model_name="distilbert-base-uncased", max_length=256)
        pipeline = LogBERTPipeline(config)
        
        print("✅ Pipeline ready for anomaly detection training")
        
        # Create dataset for training
        dataset = pipeline.create_dataset(
            anomaly_df['log_message'], 
            anomaly_df['is_anomaly'],
            transform=True
        )
        print(f"   Dataset size: {len(dataset)}")
    
    print(f"\n🔹 SCENARIO 2: MULTI-SYSTEM LOG CORRELATION")
    print("-" * 40)
    
    # Simulate logs from different systems
    systems = ['web-server', 'database', 'cache', 'auth-service']
    
    multi_system_logs = []
    for i in range(200):
        system = np.random.choice(systems)
        timestamp = f"2024-12-24 {np.random.randint(10, 18):02d}:{np.random.randint(0, 60):02d}:{np.random.randint(0, 60):02d}"
        
        if system == 'web-server':
            log = f"{timestamp} INFO [{system}] HTTP request processed: 200 OK"
        elif system == 'database':
            log = f"{timestamp} DEBUG [{system}] Query executed in 45ms"
        elif system == 'cache':
            log = f"{timestamp} INFO [{system}] Cache hit ratio: 85%"
        else:  # auth-service
            log = f"{timestamp} INFO [{system}] User authentication successful"
        
        multi_system_logs.append({
            'timestamp': timestamp,
            'system': system,
            'log_message': log
        })
    
    multi_df = pd.DataFrame(multi_system_logs)
    print(f"Multi-system dataset: {len(multi_df)} logs from {len(systems)} systems")
    print(f"System distribution: {multi_df['system'].value_counts().to_dict()}")
    
    print(f"\n🔹 SCENARIO 3: REAL-TIME STREAMING")
    print("-" * 40)
    
    print("Streaming log processing simulation:")
    
    # Simulate streaming logs
    stream_logs = generate_sample_dataset(50)['log_message'].tolist()
    
    if TRANSFORMERS_AVAILABLE:
        config = BERTConfig(batch_size=8, max_length=128)
        streaming_pipeline = LogBERTPipeline(config)
        
        # Process in mini-batches (streaming simulation)
        batch_size = 10
        for i in range(0, len(stream_logs), batch_size):
            batch = stream_logs[i:i + batch_size]
            print(f"   Processing stream batch {i//batch_size + 1}: {len(batch)} logs")
            
            # In real scenario, you'd process these immediately
            if i >= 20:  # Limit demo
                break
    
    print(f"\n🔹 BEST PRACTICES SUMMARY")
    print("-" * 40)
    
    best_practices = [
        "✅ Use appropriate batch sizes (16-32 for BERT)",
        "✅ Preprocess logs before BERT tokenization",
        "✅ Monitor memory usage with large datasets",
        "✅ Save processed batches for reuse",
        "✅ Use DataLoader with multiple workers for speed",
        "✅ Implement proper error handling",
        "✅ Version control your pipeline configurations",
        "✅ Monitor token length distribution",
        "✅ Use appropriate BERT model size for your use case"
    ]
    
    for practice in best_practices:
        print(f"   {practice}")


def main():
    """
    Main demonstration function.
    """
    
    print("🚀 BERT-READY LOG PROCESSING PIPELINE")
    print("=" * 60)
    print("This demonstration shows how to process log datasets for BERT embedding")
    print("and ML applications with proper batching and data handling.")
    
    if not TRANSFORMERS_AVAILABLE:
        print("\n⚠️  Note: Some features require transformers library")
        print("📦 Install with: pip install transformers torch tqdm")
    
    # Run demonstrations
    demonstrate_input_output_formats()
    demonstrate_batching_strategies()
    demonstrate_ml_integration()
    demonstrate_real_world_usage()
    
    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    print("\n📚 Key Features Demonstrated:")
    print("   🔤 Advanced log tokenization with special token handling")
    print("   🤖 BERT-ready preprocessing pipeline")
    print("   📦 Efficient batching strategies for large datasets")
    print("   🔥 PyTorch Dataset and DataLoader integration")
    print("   🧪 Scikit-learn pipeline compatibility")
    print("   🎯 Classification and anomaly detection setups")
    print("   💾 Data persistence and loading capabilities")
    print("   📊 Comprehensive input/output format support")
    
    print("\n🎯 Ready for:")
    print("   • Log classification and categorization")
    print("   • Anomaly detection in system logs")
    print("   • Log similarity and clustering")
    print("   • Multi-system log correlation")
    print("   • Real-time log analysis")
    print("   • Custom BERT fine-tuning on log data")


if __name__ == "__main__":
    main() 