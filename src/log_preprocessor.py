"""
Log Preprocessing Function for AI-based Log Filtering
====================================================

This module provides functions to clean and preprocess log entries for machine learning.
Each preprocessing step is explained with its importance for ML models.
Works with both individual log messages and pandas DataFrames.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union


def preprocess_logs_dataframe(df: pd.DataFrame, log_column: str = 'log_message') -> pd.DataFrame:
    """
    Preprocess a DataFrame of log entries for machine learning.
    
    Args:
        df (pd.DataFrame): DataFrame containing log entries
        log_column (str): Name of the column containing log messages
        
    Returns:
        pd.DataFrame: Enhanced DataFrame with preprocessing results
    """
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Apply preprocessing to each log entry
    preprocessing_results = result_df[log_column].apply(clean_log_message)
    
    # Extract results into separate columns
    result_df['cleaned_message'] = preprocessing_results.apply(lambda x: x['cleaned'])
    result_df['preserved_tokens'] = preprocessing_results.apply(lambda x: x['preserved_tokens'])
    result_df['removed_timestamp'] = preprocessing_results.apply(lambda x: x['removed_elements']['timestamp'])
    
    # Extract specific token types into separate columns for ML features
    result_df['has_ip_address'] = preprocessing_results.apply(lambda x: 'ip_addresses' in x['preserved_tokens'])
    result_df['has_file_path'] = preprocessing_results.apply(lambda x: 'file_paths' in x['preserved_tokens'])
    result_df['has_error_code'] = preprocessing_results.apply(lambda x: 'error_codes' in x['preserved_tokens'])
    result_df['has_url'] = preprocessing_results.apply(lambda x: 'urls' in x['preserved_tokens'])
    result_df['has_email'] = preprocessing_results.apply(lambda x: 'emails' in x['preserved_tokens'])
    
    # Extract log level if present
    result_df['log_level'] = result_df[log_column].apply(_extract_log_level)
    
    # Calculate text statistics
    result_df['original_length'] = result_df[log_column].str.len()
    result_df['cleaned_length'] = result_df['cleaned_message'].str.len()
    result_df['word_count'] = result_df['cleaned_message'].str.split().str.len()
    
    return result_df


def create_ml_features_dataframe(df: pd.DataFrame, text_column: str = 'cleaned_message') -> pd.DataFrame:
    """
    Create additional ML features from preprocessed log data.
    
    Args:
        df (pd.DataFrame): DataFrame with preprocessed logs
        text_column (str): Column containing cleaned text
        
    Returns:
        pd.DataFrame: DataFrame with additional ML features
    """
    
    feature_df = df.copy()
    
    # Text-based features
    feature_df['char_count'] = feature_df[text_column].str.len()
    feature_df['word_count'] = feature_df[text_column].str.split().str.len()
    feature_df['avg_word_length'] = feature_df[text_column].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
    )
    
    # Pattern-based features
    feature_df['contains_numbers'] = feature_df[text_column].str.contains(r'\d+', regex=True)
    feature_df['contains_uppercase'] = feature_df[text_column].str.contains(r'[A-Z]', regex=True)
    feature_df['special_char_count'] = feature_df[text_column].apply(
        lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x))
    )
    
    # Common log keywords
    log_keywords = ['error', 'warning', 'info', 'debug', 'critical', 'fatal', 'success', 'failed', 'timeout']
    for keyword in log_keywords:
        feature_df[f'contains_{keyword}'] = feature_df[text_column].str.contains(keyword, case=False, regex=False)
    
    return feature_df


def load_logs_from_file(file_path: str, column_name: str = 'log_message') -> pd.DataFrame:
    """
    Load log entries from a file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the log file
        column_name (str): Name for the log message column
        
    Returns:
        pd.DataFrame: DataFrame containing log entries
    """
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logs = [line.strip() for line in f.readlines() if line.strip()]
        
        df = pd.DataFrame({column_name: logs})
        df['entry_id'] = range(len(df))
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()


def clean_log_message(raw_log: str) -> Dict[str, str]:
    """
    Clean a raw log message for machine learning processing.
    
    This function performs several preprocessing steps:
    1. Extract and preserve meaningful tokens (before text transformation)
    2. Remove timestamps
    3. Extract log level (before removing it from text)
    4. Convert to lowercase
    5. Remove log level from text (to prevent data leakage)
    6. Remove special characters and extra whitespace
    7. Preserve structure of meaningful identifiers
    
    Args:
        raw_log (str): Raw log entry
        
    Returns:
        Dict containing:
            - 'original': Original log message
            - 'cleaned': Cleaned message ready for ML
            - 'preserved_tokens': Important tokens extracted
            - 'removed_elements': Elements that were removed
    """
    
    # Step 1: Extract and preserve meaningful tokens BEFORE transformation
    preserved_tokens = _extract_meaningful_tokens(raw_log)
    
    # Step 2: Remove timestamps
    cleaned, removed_timestamp = _remove_timestamps(raw_log)
    
    # Step 3: Extract log level BEFORE converting to lowercase
    log_level = _extract_log_level(cleaned)
    
    # Step 4: Convert to lowercase (after preserving case-sensitive tokens)
    cleaned = _convert_to_lowercase(cleaned)
    
    # Step 5: Remove log level from cleaned text (CRITICAL for ML)
    cleaned = _remove_log_level(cleaned, log_level)
    
    # Step 6: Remove special characters while preserving structure
    cleaned, removed_chars = _remove_special_characters(cleaned)
    
    # Step 7: Clean whitespace
    cleaned = _clean_whitespace(cleaned)
    
    # Step 8: Replace preserved tokens with normalized placeholders
    cleaned = _replace_with_placeholders(cleaned, preserved_tokens)
    
    return {
        'original': raw_log,
        'cleaned': cleaned,
        'preserved_tokens': preserved_tokens,
        'removed_elements': {
            'timestamp': removed_timestamp,
            'log_level': log_level,
            'special_chars': removed_chars
        }
    }


def _extract_log_level(text: str) -> str:
    """Extract log level from log entry."""
    log_levels = ['CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'TRACE']
    
    for level in log_levels:
        if re.search(rf'\b{level}\b', text, re.IGNORECASE):
            return level.upper()
    return 'UNKNOWN'


def _extract_meaningful_tokens(text: str) -> Dict[str, List[str]]:
    """
    Extract meaningful tokens that should be preserved for ML analysis.
    
    WHY THIS IS IMPORTANT:
    - Error codes, IPs, and paths contain crucial diagnostic information
    - These patterns often indicate specific system states or failures
    - Preserving them maintains semantic meaning for classification
    """
    
    patterns = {
        # IP addresses - critical for security analysis
        'ip_addresses': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        
        # File paths - important for system analysis
        'file_paths': r'(?:[a-zA-Z]:\\[^\s]+|/[^\s]+\.[\w]+|/[^\s]*(?:/[^\s]+)+)',
        
        # Error codes - crucial for error classification
        'error_codes': r'\b(?:[A-Z]{2,}\d{3,}|\d{3,5}|E\d+|ERR_\w+)\b',
        
        # URLs - important for web log analysis
        'urls': r'https?://[^\s<>"{}|\\^`\[\]]+',
        
        # UUIDs and hashes - unique identifiers
        'identifiers': r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
        
        # Email addresses
        'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        
        # Memory addresses and hex values
        'hex_values': r'\b0x[0-9a-fA-F]+\b',
        
        # Port numbers
        'ports': r':\d{2,5}\b',
    }
    
    extracted = {}
    for token_type, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            extracted[token_type] = matches
            
    return extracted


def _remove_timestamps(text: str) -> Tuple[str, str]:
    """
    Remove timestamp information from log entries.
    
    WHY THIS IS IMPORTANT:
    - Timestamps are usually not relevant for log classification
    - They add noise and increase feature dimensionality
    - Time-based features should be extracted separately if needed
    - Removing them helps models focus on the actual log content
    """
    
    timestamp_patterns = [
        # ISO format: 2024-08-11 12:22:59
        r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}(?:\.\d+)?',
        
        # US format: 08/11/2024 12:22:59
        r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}',
        
        # Syslog format: Aug 11 12:22:59
        r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',
        
        # Unix timestamp
        r'\b\d{10}\b',
        
        # Bracketed timestamps: [2024-08-11T12:22:59]
        r'\[[\d\-T:\.\s]+\]',
    ]
    
    removed_timestamp = ""
    cleaned = text
    
    for pattern in timestamp_patterns:
        match = re.search(pattern, cleaned)
        if match:
            removed_timestamp = match.group()
            cleaned = re.sub(pattern, '', cleaned)
            break
    
    return cleaned.strip(), removed_timestamp


def _convert_to_lowercase(text: str) -> str:
    """
    Convert text to lowercase for normalization.
    
    WHY THIS IS IMPORTANT:
    - Reduces vocabulary size and feature dimensionality
    - "ERROR", "Error", and "error" should be treated as the same token
    - Improves model generalization across different log formats
    - Essential for consistent text processing in NLP pipelines
    """
    return text.lower()


def _remove_special_characters(text: str) -> Tuple[str, List[str]]:
    """
    Remove special characters while preserving meaningful structure.
    
    WHY THIS IS IMPORTANT:
    - Special characters often don't contribute to semantic meaning
    - They increase feature space without adding value
    - However, some punctuation in paths/codes should be preserved
    - Reduces noise for text-based ML algorithms
    """
    
    # Characters to remove (but preserve structure in meaningful tokens)
    removed_chars = []
    
    # Keep alphanumeric, spaces, and some meaningful punctuation
    # Remove brackets, quotes, excessive punctuation
    cleaned = re.sub(r'[^\w\s\.\-/_:@]', ' ', text)
    
    # Track what was removed for analysis
    original_chars = set(text)
    cleaned_chars = set(cleaned)
    removed_chars = list(original_chars - cleaned_chars)
    
    return cleaned, removed_chars


def _clean_whitespace(text: str) -> str:
    """
    Clean up whitespace issues.
    
    WHY THIS IS IMPORTANT:
    - Inconsistent whitespace creates different tokens for same content
    - Multiple spaces should be normalized to single spaces
    - Leading/trailing whitespace adds no semantic value
    - Consistent spacing improves tokenization quality
    """
    
    # Replace multiple whitespace with single space
    cleaned = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def _replace_with_placeholders(text: str, preserved_tokens: Dict[str, List[str]]) -> str:
    """
    Replace preserved tokens with normalized placeholders.
    
    WHY THIS IS IMPORTANT:
    - Maintains semantic information while reducing vocabulary
    - IP "192.168.1.1" and "10.0.0.1" both become "<IP_ADDRESS>"
    - Helps model learn patterns rather than memorizing specific values
    - Improves model generalization to new, unseen data
    """
    
    cleaned = text
    
    # Define placeholder mappings
    placeholder_map = {
        'ip_addresses': '<IP_ADDRESS>',
        'file_paths': '<FILE_PATH>',
        'error_codes': '<ERROR_CODE>',
        'urls': '<URL>',
        'identifiers': '<IDENTIFIER>',
        'emails': '<EMAIL>',
        'hex_values': '<HEX_VALUE>',
        'ports': '<PORT>',
    }
    
    # Replace each type of token with its placeholder
    for token_type, tokens in preserved_tokens.items():
        if token_type in placeholder_map:
            placeholder = placeholder_map[token_type]
            for token in tokens:
                cleaned = cleaned.replace(token.lower(), placeholder)
    
    return cleaned


def _remove_log_level(text: str, log_level: str) -> str:
    """
    Remove log level from the cleaned text.
    
    WHY THIS IS CRITICAL:
    - Prevents data leakage in ML models
    - Forces models to learn from actual log content, not just log level keywords
    - Log level is extracted as a separate feature for supervised learning
    - Essential for building robust classification models
    """
    
    if log_level and log_level != 'UNKNOWN':
        # Remove the log level word from text (case insensitive)
        pattern = rf'\b{re.escape(log_level.lower())}\b'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Also remove common variations
        level_variations = {
            'WARNING': ['warn'],
            'CRITICAL': ['crit'],
            'FATAL': ['fatal']
        }
        
        if log_level in level_variations:
            for variation in level_variations[log_level]:
                pattern = rf'\b{re.escape(variation)}\b'
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


# Demonstration and testing function
def demonstrate_dataframe_preprocessing():
    """
    Demonstrate the DataFrame preprocessing with various log examples.
    """
    
    # Create sample DataFrame
    sample_logs = [
        "2024-08-11 12:22:59 INFO User john.doe@company.com logged in from 192.168.1.100",
        "2024-08-11 12:23:01 DEBUG Fetching user details from /var/log/users.db",
        "Aug 11 12:23:03 ERROR Failed to connect to https://api.service.com:8443 - Error code E5001",
        "[2024-08-11T12:23:05] CRITICAL Memory allocation failed at 0x7fff5fbff7e0",
        "12:23:07 WARN File /tmp/cache/user_123.tmp not found (errno: 2)"
    ]
    
    # Create DataFrame
    df = pd.DataFrame({'log_message': sample_logs})
    
    print("=" * 80)
    print("DATAFRAME LOG PREPROCESSING DEMONSTRATION")
    print("=" * 80)
    
    # Process the DataFrame
    processed_df = preprocess_logs_dataframe(df)
    
    # Display results
    print("\nOriginal DataFrame:")
    print(df)
    
    print("\nProcessed DataFrame (key columns):")
    key_columns = ['log_message', 'cleaned_message', 'log_level', 'has_ip_address', 'has_error_code']
    print(processed_df[key_columns])
    
    print("\nML Features DataFrame:")
    ml_features_df = create_ml_features_dataframe(processed_df)
    feature_columns = ['cleaned_message', 'word_count', 'contains_error', 'contains_failed', 'has_ip_address']
    print(ml_features_df[feature_columns])
    
    return processed_df


if __name__ == "__main__":
    # Run DataFrame demonstration
    demonstrate_dataframe_preprocessing()
    
    # Example of loading from file
    print("\n" + "=" * 80)
    print("LOADING FROM FILE EXAMPLE")
    print("=" * 80)
    
    # Try to load and process logs.txt
    df = load_logs_from_file('../logs.txt')
    if not df.empty:
        print("Loaded logs:")
        print(df)
        
        print("\nProcessed logs:")
        processed = preprocess_logs_dataframe(df)
        print(processed[['log_message', 'cleaned_message', 'log_level']])
    else:
        print("Could not load logs.txt - using sample data instead") 