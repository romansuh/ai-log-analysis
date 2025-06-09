from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
import logging
from bert_vectorization_pipeline import process_logs_with_vectors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogNaiveBayesClassifier:
    def __init__(self):
        self.classifier = GaussianNB()
        
    def train(self, df: pd.DataFrame):
        """Train the classifier using BERT vectors and cluster_id."""
        # Extract BERT vectors and convert to numpy array
        X = np.vstack(df['bert_vector'].values)
        y = df['cluster_id'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train classifier
        logger.info("Training Naive Bayes classifier...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(report)
        
        return X_test, y_test
    
    def predict(self, vectors):
        """Predict clusters for new BERT vectors."""
        return self.classifier.predict(vectors)

def main():
    """Demonstrate Naive Bayes classification on log vectors."""
    print("Processing logs with BERT vectors...")
    
    # Use the same pipeline as example_usage.py
    df = process_logs_with_vectors('Zookeeper_2k.log')
    
    print(f"✓ Processed {len(df)} log entries")
    print(f"✓ Vector dimension: {len(df['bert_vector'].iloc[0])}")
    
    # Initialize and train classifier
    classifier = LogNaiveBayesClassifier()
    X_test, y_test = classifier.train(df)
    
    # Make predictions on test set
    y_pred = classifier.predict(X_test)
    
    # Show some example predictions with messages
    logger.info("\nExample predictions:")
    test_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    for idx in test_indices:
        logger.info(f"Message: {df['message'].iloc[idx]}")
        logger.info(f"True cluster: {y_test[idx]}, Predicted cluster: {y_pred[idx]}\n")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("Error: Zookeeper_2k.log not found")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install: pip install -r requirements.txt") 