# BERT-Ready Log Processing Pipeline Guide

## Overview

This guide explains the comprehensive BERT-ready log processing pipeline that builds on our advanced tokenization system. The pipeline provides scikit-learn compatibility, efficient batching, and seamless integration with PyTorch and Hugging Face transformers for state-of-the-art log analysis.

## 🎯 Key Features

### 🔄 **Complete Data Pipeline**
- **Input flexibility**: DataFrames, lists, arrays, CSV files
- **Output formats**: BERT tensors, PyTorch datasets, batched data
- **Scikit-learn compatibility**: Drop-in replacement for sklearn transformers

### 🤖 **BERT Integration** 
- **Pre-trained models**: Support for any Hugging Face transformer
- **Optimized tokenization**: Handles log-specific preprocessing before BERT
- **Batch processing**: Memory-efficient processing for large datasets

### ⚡ **Performance & Scalability**
- **Streaming processing**: Handle datasets too large for memory
- **Adaptive batching**: Optimize batch sizes based on text length
- **Multi-worker support**: Parallel processing with PyTorch DataLoader

## 📦 Installation & Setup

### Dependencies

```bash
# Install required packages
pip install transformers torch tqdm

# Or install all requirements
pip install -r requirements.txt
```

### Quick Start

```python
from src.bert_pipeline import LogBERTPipeline, BERTConfig
import pandas as pd

# Load your log data
df = pd.read_csv('your_logs.csv')

# Configure pipeline
config = BERTConfig(
    model_name="bert-base-uncased",
    max_length=512,
    batch_size=32
)

# Initialize and fit pipeline
pipeline = LogBERTPipeline(config)
pipeline.fit(df['log_message'])

# Transform to BERT-ready format
bert_tensors = pipeline.transform(df['log_message'])
```

## 🔧 Configuration

### BERTConfig Options

```python
@dataclass
class BERTConfig:
    model_name: str = "bert-base-uncased"      # Any HuggingFace model
    max_length: int = 512                      # Maximum sequence length
    padding: str = "max_length"                # Padding strategy
    truncation: bool = True                    # Truncate long sequences
    return_tensors: str = "pt"                 # Return PyTorch tensors
    batch_size: int = 32                       # Default batch size
    use_special_tokens: bool = True            # Add [CLS], [SEP] tokens
```

### Recommended Configurations

#### 🎯 **Production Log Analysis**
```python
config = BERTConfig(
    model_name="bert-base-uncased",
    max_length=512,
    batch_size=16,  # Adjust based on GPU memory
    padding="max_length",
    truncation=True
)
```

#### 🚀 **Fast Processing (Smaller Model)**
```python
config = BERTConfig(
    model_name="distilbert-base-uncased",
    max_length=256,
    batch_size=64,
    padding="max_length"
)
```

#### 🔬 **Research/Experimentation**
```python
config = BERTConfig(
    model_name="roberta-base",
    max_length=512,
    batch_size=8,
    padding="longest",  # Dynamic padding
    truncation=True
)
```

## 📊 Input/Output Formats

### Supported Input Formats

#### 1. **Pandas DataFrame** (Recommended)
```python
# Auto-detects 'log_message' or 'message' column
df = pd.DataFrame({
    'timestamp': timestamps,
    'log_message': log_texts,
    'log_level': levels
})

pipeline.fit(df)
```

#### 2. **List of Strings**
```python
log_texts = [
    "2024-12-24 INFO User logged in successfully",
    "2024-12-24 ERROR Database connection failed",
    # ... more logs
]

pipeline.fit(log_texts)
```

#### 3. **NumPy Array**
```python
import numpy as np

log_array = np.array(log_texts)
pipeline.fit(log_array)
```

#### 4. **CSV File**
```python
# Load and process CSV
df = pd.read_csv('logs.csv')
pipeline.fit(df['log_message'])
```

### Output Formats

#### 1. **BERT Tensors**
```python
result = pipeline.transform(texts)
# Returns: {
#     'input_ids': torch.Tensor,      # Shape: (batch_size, max_length)
#     'attention_mask': torch.Tensor  # Shape: (batch_size, max_length)
# }

input_ids = result['input_ids']        # Token IDs
attention_mask = result['attention_mask']  # Attention mask
```

#### 2. **PyTorch Dataset**
```python
dataset = pipeline.create_dataset(texts, labels=labels)

# Access individual items
item = dataset[0]
# Returns: {
#     'input_ids': torch.Tensor,
#     'attention_mask': torch.Tensor,
#     'text': str,
#     'labels': torch.Tensor  # if labels provided
# }
```

#### 3. **DataLoader (Batched)**
```python
dataloader = pipeline.create_dataloader(
    dataset, 
    batch_size=32, 
    shuffle=True
)

for batch in dataloader:
    input_ids = batch['input_ids']          # (batch_size, max_length)
    attention_mask = batch['attention_mask'] # (batch_size, max_length)
    labels = batch['labels']                # (batch_size,) if available
    texts = batch['texts']                  # List of original texts
```

#### 4. **Saved Batches**
```python
# Process and save large datasets
for batch in pipeline.process_dataset_in_batches(
    large_dataset, 
    batch_size=1000,
    save_path="processed_batches/"
):
    # Batches automatically saved as .pt files
    pass

# Load saved batch
batch = torch.load("processed_batches/batch_0000.pt")
```

## 🔄 Pipeline Components

### 1. **LogPreprocessor** (Scikit-learn Compatible)

```python
from src.bert_pipeline import LogPreprocessor

preprocessor = LogPreprocessor(
    preserve_special_tokens=True,
    use_lemmatization=True,
    remove_stopwords=True,
    min_token_length=2,
    join_tokens=True  # Join back into string for BERT
)

# Fit and transform
preprocessor.fit(raw_logs)
processed_texts = preprocessor.transform(raw_logs)
```

### 2. **BERTTokenizer** (Scikit-learn Compatible)

```python
from src.bert_pipeline import BERTTokenizer

bert_tokenizer = BERTTokenizer(
    model_name="bert-base-uncased",
    max_length=512,
    padding="max_length",
    truncation=True
)

# Fit and transform
bert_tokenizer.fit(processed_texts)
bert_tensors = bert_tokenizer.transform(processed_texts)
```

### 3. **Complete Pipeline**

```python
# Combines both components
pipeline = LogBERTPipeline(config)

# Single step: raw logs → BERT tensors
bert_output = pipeline.fit_transform(raw_logs)
```

## 📦 Batching Strategies

### 1. **Memory-Efficient Streaming**

For datasets too large for memory:

```python
# Process in chunks without loading entire dataset
for batch in pipeline.process_dataset_in_batches(
    huge_dataset, 
    batch_size=1000,
    save_path="batches/"
):
    # Process each batch individually
    # Memory usage stays constant
    process_batch(batch)
```

### 2. **DataLoader with Workers**

For parallel processing:

```python
dataset = pipeline.create_dataset(texts, labels)
dataloader = pipeline.create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4  # Parallel loading
)
```

### 3. **Adaptive Batching**

Optimize batch sizes based on text length:

```python
# Sort by length for efficient batching
df_sorted = df.sort_values(df['log_message'].str.len())

# Short texts: larger batches
short_texts = df_sorted[df_sorted['log_message'].str.len() < 100]
short_loader = pipeline.create_dataloader(
    pipeline.create_dataset(short_texts['log_message']),
    batch_size=64
)

# Long texts: smaller batches
long_texts = df_sorted[df_sorted['log_message'].str.len() >= 200]
long_loader = pipeline.create_dataloader(
    pipeline.create_dataset(long_texts['log_message']),
    batch_size=16
)
```

## 🤖 ML Integration Examples

### 1. **Log Classification**

```python
from src.bert_pipeline import LogClassificationPipeline

# Setup classification pipeline
clf_pipeline = LogClassificationPipeline(config)
clf_pipeline.fit(df['log_message'], df['log_level'])

# Prepare train/validation split
train_loader, val_loader = clf_pipeline.prepare_train_val_split(
    df['log_message'], 
    df['log_level'],
    test_size=0.2,
    stratify=True
)

# Ready for training with any PyTorch model
for batch in train_loader:
    # Train your model
    outputs = model(batch['input_ids'], batch['attention_mask'])
    loss = criterion(outputs, batch['labels'])
    # ... training loop
```

### 2. **Anomaly Detection**

```python
# Binary classification setup
df['is_anomaly'] = (df['log_level'] == 'CRITICAL').astype(int)

dataset = pipeline.create_dataset(
    df['log_message'], 
    df['is_anomaly']
)

dataloader = pipeline.create_dataloader(dataset, batch_size=32)

# Train anomaly detection model
for batch in dataloader:
    # Binary classification
    outputs = anomaly_model(batch['input_ids'], batch['attention_mask'])
    loss = bce_loss(outputs, batch['labels'].float())
```

### 3. **Scikit-learn Integration**

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Create sklearn pipeline with log preprocessing
sklearn_pipeline = Pipeline([
    ('log_preprocessor', LogPreprocessor()),
    ('vectorizer', TfidfVectorizer()),  # Or use BERT features
    ('classifier', LogisticRegression())
])

# Fit and predict
sklearn_pipeline.fit(df['log_message'], df['log_level'])
predictions = sklearn_pipeline.predict(new_logs)
```

### 4. **Custom BERT Model Integration**

```python
import torch.nn as nn
from transformers import AutoModel

class LogBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# Use with our pipeline
model = LogBERTClassifier("bert-base-uncased", num_classes=5)

for batch in train_loader:
    outputs = model(batch['input_ids'], batch['attention_mask'])
    loss = criterion(outputs, batch['labels'])
```

## 🎯 Real-World Use Cases

### 1. **Log Anomaly Detection**

```python
# Setup
config = BERTConfig(model_name="distilbert-base-uncased", max_length=256)
pipeline = LogBERTPipeline(config)

# Prepare anomaly dataset
normal_logs = load_normal_logs()
anomalous_logs = load_anomalous_logs()

df = pd.DataFrame({
    'log_message': normal_logs + anomalous_logs,
    'is_anomaly': [0] * len(normal_logs) + [1] * len(anomalous_logs)
})

# Create datasets
dataset = pipeline.create_dataset(df['log_message'], df['is_anomaly'])
train_loader = pipeline.create_dataloader(dataset, batch_size=32)

# Train anomaly detection model
# ... training code
```

### 2. **Multi-System Log Correlation**

```python
# Logs from different systems
systems = ['web-server', 'database', 'cache', 'auth-service']

# Encode system as additional feature
df['system_encoded'] = pd.Categorical(df['system']).codes

# Custom dataset with system information
class MultiSystemDataset(LogDataset):
    def __init__(self, texts, systems, **kwargs):
        super().__init__(texts, **kwargs)
        self.systems = systems
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['system'] = torch.tensor(self.systems[idx])
        return item

dataset = MultiSystemDataset(
    df['log_message'], 
    df['system_encoded'],
    tokenizer=pipeline.bert_tokenizer,
    max_length=config.max_length
)
```

### 3. **Real-Time Log Processing**

```python
import asyncio
from collections import deque

class StreamingLogProcessor:
    def __init__(self, pipeline, batch_size=32):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.buffer = deque()
    
    async def process_log(self, log_message):
        self.buffer.append(log_message)
        
        if len(self.buffer) >= self.batch_size:
            # Process batch
            batch = list(self.buffer)
            self.buffer.clear()
            
            # Process with BERT pipeline
            result = self.pipeline.transform(batch)
            
            # Send to downstream processing
            await self.handle_batch(result)
    
    async def handle_batch(self, bert_output):
        # Implement your real-time processing logic
        pass

# Usage
processor = StreamingLogProcessor(pipeline)
await processor.process_log("2024-12-24 ERROR Connection failed")
```

## 📈 Performance Optimization

### Memory Management

```python
# Monitor memory usage
import torch

if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

# Clear cache when needed
torch.cuda.empty_cache()

# Use smaller batch sizes for large models
config = BERTConfig(
    model_name="bert-large-uncased",
    batch_size=8  # Reduce for large models
)
```

### Batch Size Optimization

```python
# Estimate optimal batch size
def find_optimal_batch_size(pipeline, sample_texts, max_batch_size=128):
    for batch_size in [8, 16, 32, 64, max_batch_size]:
        try:
            dataset = pipeline.create_dataset(sample_texts[:batch_size])
            dataloader = pipeline.create_dataloader(dataset, batch_size=batch_size)
            
            # Test one batch
            batch = next(iter(dataloader))
            del batch
            
            print(f"Batch size {batch_size}: OK")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: OOM")
                return batch_size // 2
            else:
                raise e
    
    return max_batch_size
```

### Model Selection

```python
# Model size vs accuracy trade-offs
models = {
    "distilbert-base-uncased": {"size": "66M", "speed": "fast", "accuracy": "good"},
    "bert-base-uncased": {"size": "110M", "speed": "medium", "accuracy": "better"},
    "bert-large-uncased": {"size": "340M", "speed": "slow", "accuracy": "best"},
    "roberta-base": {"size": "125M", "speed": "medium", "accuracy": "better+"}
}

# Choose based on your requirements
config = BERTConfig(
    model_name="distilbert-base-uncased",  # Good balance for logs
    max_length=256,  # Most logs are shorter
    batch_size=32
)
```

## 💾 Persistence & Deployment

### Save/Load Pipeline

```python
# Save trained pipeline
pipeline.save_pipeline("models/log_bert_pipeline.pkl")

# Load pipeline
loaded_pipeline = LogBERTPipeline.load_pipeline("models/log_bert_pipeline.pkl")

# Use loaded pipeline
result = loaded_pipeline.transform(new_logs)
```

### Export for Production

```python
# Export processed datasets
processed_data = {
    'train': train_loader,
    'val': val_loader,
    'config': config,
    'statistics': pipeline.statistics_
}

torch.save(processed_data, "production/log_dataset.pt")

# Load in production
production_data = torch.load("production/log_dataset.pt")
```

### Docker Deployment

```dockerfile
# Dockerfile for BERT log processing
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

CMD ["python", "-m", "src.bert_pipeline"]
```

## 🔧 Troubleshooting

### Common Issues

#### 1. **Out of Memory Errors**
```python
# Solutions:
# - Reduce batch_size
# - Use gradient checkpointing
# - Switch to smaller model
# - Use CPU instead of GPU

config = BERTConfig(
    model_name="distilbert-base-uncased",  # Smaller model
    batch_size=8,  # Smaller batches
    max_length=256  # Shorter sequences
)
```

#### 2. **Slow Processing**
```python
# Solutions:
# - Increase batch_size (if memory allows)
# - Use DataLoader with multiple workers
# - Cache preprocessed data

dataloader = pipeline.create_dataloader(
    dataset,
    batch_size=64,  # Larger batches
    num_workers=4   # Parallel loading
)
```

#### 3. **Token Length Issues**
```python
# Monitor token lengths
stats = pipeline.statistics_
print(f"Max token length: {stats['max_token_length']}")
print(f"Tokens exceeding limit: {stats['tokens_exceeding_limit']}")

# Adjust max_length if needed
if stats['tokens_exceeding_limit'] > stats['total_samples'] * 0.1:
    print("Consider increasing max_length")
```

### Debugging Tools

```python
# Debug pipeline step by step
result = pipeline.transform(["Sample log message"])

print("Input IDs shape:", result['input_ids'].shape)
print("Attention mask shape:", result['attention_mask'].shape)
print("First few tokens:", result['input_ids'][0][:10])

# Decode tokens back to text
decoded = pipeline.bert_tokenizer.decode(result['input_ids'][0])
print("Decoded text:", decoded)
```

## 🎓 Best Practices

### ✅ **Do's**

1. **Preprocess logs before BERT tokenization**
   - Use our LogPreprocessor for log-specific cleaning
   - Preserve special tokens for semantic meaning

2. **Monitor resource usage**
   - Start with smaller batch sizes and increase gradually
   - Monitor GPU memory usage
   - Use mixed precision training when possible

3. **Validate your pipeline**
   - Check token length distributions
   - Verify special token preservation
   - Test on representative data samples

4. **Use appropriate model sizes**
   - DistilBERT for speed-critical applications
   - BERT-base for balanced performance
   - BERT-large only when accuracy is paramount

5. **Implement proper error handling**
   - Handle OOM errors gracefully
   - Validate input formats
   - Log processing statistics

### ❌ **Don'ts**

1. **Don't skip log preprocessing**
   - Raw logs contain noise that hurts BERT performance
   - Always use domain-specific preprocessing

2. **Don't use excessive sequence lengths**
   - Most logs are short; 256-512 tokens usually sufficient
   - Longer sequences = quadratic memory usage

3. **Don't ignore batch size optimization**
   - Too small = inefficient GPU usage
   - Too large = OOM errors
   - Find the sweet spot for your hardware

4. **Don't forget about inference performance**
   - Consider model quantization for production
   - Use ONNX for faster inference
   - Cache frequently processed logs

This comprehensive pipeline provides everything needed for state-of-the-art log analysis with BERT while maintaining efficiency and scalability for real-world applications. 