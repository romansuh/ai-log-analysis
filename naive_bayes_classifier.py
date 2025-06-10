from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from bert_vectorization_pipeline import process_logs_with_vectors
from cluster_visualization import add_kmeans_clusters
from datetime import datetime
import io

class LogNaiveBayesClassifier:
    def __init__(self):
        self.classifier = GaussianNB()
        
    def train(self, df: pd.DataFrame, cluster_field_name='KMeans_cluster_id'):
        """
        Train the classifier using normalized BERT vectors and KMeans cluster_id.
        
        Args:
            df: DataFrame containing BERT vectors and KMeans_cluster_id
            
        Returns:
            Test data and predictions
        """
        # Extract BERT vectors and normalize them
        X = normalize(np.vstack(df['bert_vector'].values))
        y = df[cluster_field_name].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train classifier
        print("Training Naive Bayes classifier...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_test, y_test, y_pred, report
    
    def predict(self, vectors):
        """
        Predict clusters for new BERT vectors.
        
        Args:
            vectors: BERT vectors to predict clusters for
            
        Returns:
            Predicted cluster labels
        """
        # Normalize input vectors
        normalized_vectors = normalize(vectors)
        return self.classifier.predict(normalized_vectors)

def save_classification_report_to_csv(report, accuracy, cluster_field_name: str):
    """Save classification report metrics to a CSV file."""
    # Convert report to DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    # Get the actual class numbers from the report (excluding accuracy and averages)
    class_numbers = [int(idx) for idx in report_df.index if idx not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Create a new DataFrame with all classes from the report
    full_report = pd.DataFrame(index=class_numbers, columns=report_df.columns)
    
    # Fill in the metrics for each class
    for idx in class_numbers:
        full_report.loc[idx] = report_df.loc[str(idx)]
    
    # Add accuracy and averages back
    full_report.loc['accuracy'] = {
        'precision': accuracy,
        'recall': accuracy,
        'f1-score': accuracy,
        'support': report_df.loc['accuracy', 'support']
    }
    
    # Add macro and weighted averages
    full_report.loc['macro avg'] = report_df.loc['macro avg']
    full_report.loc['weighted avg'] = report_df.loc['weighted avg']
    
    # Add timestamp to filename
    filename = f'classification_metrics_{cluster_field_name}.csv'
    
    # Save to CSV
    full_report.to_csv(filename)
    print(f"\nClassification metrics saved to {filename}")
    print(f"Total classes in report: {len(class_numbers)}")
    print(f"Classes present: {sorted(class_numbers)}")

def main():
    """Demonstrate Naive Bayes classification on log vectors."""
    print("Processing logs with BERT vectors...")
    
    # Use the same pipeline as example_usage.py
    df = process_logs_with_vectors('Zookeeper_2k.log')
    
    print(f"✓ Processed {len(df)} log entries")
    print(f"✓ Vector dimension: {len(df['bert_vector'].iloc[0])}")
    
    # Add KMeans clusters based on BERT vector similarity
    print("\nAdding KMeans clusters based on BERT vector similarity...")
    df = add_kmeans_clusters(df, n_clusters=44)
    
    # Initialize and train classifier
    # cluster_field_name = 'KMeans_cluster_id'
    cluster_field_name = 'Drain_cluster_id'
    
    classifier = LogNaiveBayesClassifier()
    X_test, y_test, y_pred, report = classifier.train(df, cluster_field_name)
    
    # Save classification report to CSV
    save_classification_report_to_csv(report, accuracy_score(y_test, y_pred), cluster_field_name)
    
    # Show some example predictions with messages
    print("\nExample predictions:")
    test_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    for idx in test_indices:
        print(f"Message: {df['message'].iloc[idx]}")
        print(f"True cluster: {y_test[idx]}, Predicted cluster: {y_pred[idx]}\n")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("Error: Zookeeper_2k.log not found")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install: pip install -r requirements.txt") 