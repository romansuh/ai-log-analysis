# Logs Filtration using AI - Coursework

## Project Overview
This project implements an AI-based system for filtering and analyzing log data using machine learning techniques. The system can classify log entries, detect anomalies, and extract meaningful patterns from log files.

## Project Structure
```
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── data/                    # Log data files
├── src/                     # Source code
│   ├── log_preprocessor.py  # Log data preprocessing
│   ├── feature_extractor.py # Feature engineering
│   ├── models/              # ML models
│   └── utils/               # Utility functions
├── notebooks/               # Jupyter notebooks for analysis
├── results/                 # Model outputs and visualizations
└── logs.txt                # Sample log data
```

## Features
- **Log Preprocessing**: Clean and normalize log entries
- **Feature Extraction**: Extract meaningful features from log messages
- **Anomaly Detection**: Identify unusual log patterns
- **Log Classification**: Categorize logs by severity and type
- **Visualization**: Generate insights through charts and graphs

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data (required for text processing)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 3. Run the Main Analysis
```bash
python src/main.py
```

## Machine Learning Approaches
1. **Text Processing**: NLP techniques for log message analysis
2. **Classification Models**: Random Forest, SVM, Neural Networks
3. **Clustering**: Unsupervised learning for pattern discovery
4. **Anomaly Detection**: Isolation Forest, One-Class SVM

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve and AUC
- Silhouette Score for clustering 