"""
Example Usage: Log Preprocessing for ML
======================================

This script demonstrates how to use the log preprocessing functions
in a real machine learning workflow.
"""

import pandas as pd
import sys
import os

# Add src directory to path to import our module
sys.path.append('src')
from log_preprocessor import preprocess_logs_dataframe, create_ml_features_dataframe, load_logs_from_file


def main():
    print("🚀 Log Preprocessing for Machine Learning - Example Usage")
    print("=" * 60)
    
    # Method 1: Load logs from file
    print("\n📁 Method 1: Loading logs from file")
    df = load_logs_from_file('logs.txt')
    
    if df.empty:
        print("⚠️  logs.txt not found, creating sample data...")
        # Create sample log data
        sample_logs = [
            "2024-08-11 12:22:59 INFO User admin@company.com logged in from 192.168.1.100",
            "2024-08-11 12:23:01 DEBUG Fetching user profile from /var/lib/users/profiles.db",
            "2024-08-11 12:23:03 ERROR Database connection failed - timeout after 30s",
            "2024-08-11 12:23:05 WARN Invalid SSL certificate for https://api.service.com:8443",
            "2024-08-11 12:23:07 CRITICAL Memory usage exceeded 95% - OOM killer activated",
            "2024-08-11 12:23:09 INFO User session terminated successfully"
        ]
        df = pd.DataFrame({'log_message': sample_logs})
        df['entry_id'] = range(len(df))
    
    print(f"✅ Loaded {len(df)} log entries")
    print("\nOriginal Data:")
    print(df[['entry_id', 'log_message']].head())
    
    # Method 2: Preprocess the logs
    print("\n🔄 Step 1: Preprocessing logs...")
    processed_df = preprocess_logs_dataframe(df)
    
    print("✅ Preprocessing complete!")
    print("\nProcessed Data (key columns):")
    display_columns = ['log_message', 'cleaned_message', 'log_level', 'has_ip_address', 'has_error_code']
    print(processed_df[display_columns])
    
    # Method 3: Create ML features
    print("\n🤖 Step 2: Creating ML features...")
    ml_df = create_ml_features_dataframe(processed_df)
    
    print("✅ Feature engineering complete!")
    print(f"\nDataFrame shape: {ml_df.shape}")
    print(f"Total features: {len(ml_df.columns)}")
    
    # Show feature summary
    print("\n📊 Feature Summary:")
    feature_columns = [col for col in ml_df.columns if col.startswith('contains_') or col.startswith('has_')]
    print(f"Boolean features: {len(feature_columns)}")
    for col in feature_columns[:8]:  # Show first 8
        true_count = ml_df[col].sum()
        print(f"  {col}: {true_count}/{len(ml_df)} entries")
    
    # Method 4: Show text statistics
    print("\n📈 Text Statistics:")
    text_stats = ml_df[['original_length', 'cleaned_length', 'word_count']].describe()
    print(text_stats)
    
    # Method 5: Show log level distribution
    print("\n📋 Log Level Distribution:")
    level_counts = processed_df['log_level'].value_counts()
    print(level_counts)
    
    # Method 6: Export for ML
    print("\n💾 Step 3: Preparing data for ML...")
    
    # Select features for ML (numerical and boolean only)
    ml_features = ml_df.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    
    # Remove ID columns
    ml_features = [col for col in ml_features if 'id' not in col.lower()]
    
    ml_ready_df = ml_df[['cleaned_message', 'log_level'] + ml_features]
    
    print(f"✅ ML-ready dataset created with {len(ml_features)} numerical features")
    print(f"Features: {ml_features[:10]}...")  # Show first 10
    
    # Save to CSV for further analysis
    output_file = 'data/processed_logs.csv'
    os.makedirs('data', exist_ok=True)
    ml_ready_df.to_csv(output_file, index=False)
    print(f"💾 Saved processed data to: {output_file}")
    
    return ml_ready_df


if __name__ == "__main__":
    result_df = main()
    
    print("\n" + "=" * 60)
    print("🎯 Ready for Machine Learning!")
    print("Next steps:")
    print("1. Use 'cleaned_message' for text-based ML (NLP)")
    print("2. Use numerical features for traditional ML")
    print("3. Use 'log_level' as target variable for classification")
    print("4. Apply train/test split and build your models!")
    print("=" * 60) 