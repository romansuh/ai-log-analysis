"""
Advanced Log Tokenization Pipeline
==================================

This module provides comprehensive tokenization for system log messages,
including special handling for semi-structured text, special tokens,
and appropriate text normalization techniques.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import warnings


class LogTokenizer:
    """
    Advanced tokenizer specifically designed for system log messages.
    
    Handles semi-structured text, preserves meaningful tokens, and provides
    options for stemming/lemmatization appropriate for log analysis.
    """
    
    def __init__(self, 
                 preserve_special_tokens: bool = True,
                 use_stemming: bool = False,
                 use_lemmatization: bool = True,
                 remove_stopwords: bool = True,
                 min_token_length: int = 2,
                 custom_stopwords: Optional[List[str]] = None):
        """
        Initialize the LogTokenizer.
        
        Args:
            preserve_special_tokens: Whether to preserve and normalize special tokens
            use_stemming: Whether to apply stemming (not recommended for logs)
            use_lemmatization: Whether to apply lemmatization (recommended)
            remove_stopwords: Whether to remove common English stopwords
            min_token_length: Minimum length for tokens to be kept
            custom_stopwords: Additional stopwords specific to your domain
        """
        
        self.preserve_special_tokens = preserve_special_tokens
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        
        # Download required NLTK data
        self._download_nltk_requirements()
        
        # Initialize NLTK components
        if self.use_stemming:
            self.stemmer = SnowballStemmer('english')
        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        
        # Setup stopwords
        self.stopwords = set()
        if self.remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
            if custom_stopwords:
                self.stopwords.update(custom_stopwords)
        
        # Define special token patterns for logs
        self.special_patterns = self._define_special_patterns()
        
        # Define log-specific tokenization patterns
        self.log_tokenizer = self._create_log_tokenizer()
        
        print("✅ LogTokenizer initialized with:")
        print(f"   - Preserve special tokens: {preserve_special_tokens}")
        print(f"   - Stemming: {use_stemming}")
        print(f"   - Lemmatization: {use_lemmatization}")
        print(f"   - Remove stopwords: {remove_stopwords}")
        print(f"   - Min token length: {min_token_length}")
    
    def _download_nltk_requirements(self):
        """Download required NLTK data."""
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except:
                    warnings.warn(f"Could not download NLTK data: {data}")
    
    def _define_special_patterns(self) -> Dict[str, str]:
        """
        Define regex patterns for special tokens commonly found in logs.
        
        WHY THESE PATTERNS MATTER:
        - IP addresses, paths, and IDs contain crucial diagnostic information
        - They should be preserved but normalized to reduce vocabulary size
        - Different systems use different formats, so patterns must be comprehensive
        """
        
        return {
            # Network identifiers
            'IPV4_ADDRESS': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            'IPV6_ADDRESS': r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
            'MAC_ADDRESS': r'\b[0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}\b',
            'PORT_NUMBER': r'\b(?::[1-9][0-9]{1,4})\b',
            'URL': r'https?://[^\s<>"{}|\\^`\[\]]+',
            
            # File system paths
            'UNIX_PATH': r'(?:/[^\s/]+)+/?',
            'WINDOWS_PATH': r'[A-Za-z]:\\(?:[^\s\\]+\\)*[^\s\\]*',
            'FILE_EXTENSION': r'\.[a-zA-Z0-9]{1,4}\b',
            
            # Identifiers and codes
            'UUID': r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
            'HEX_VALUE': r'\b0x[0-9a-fA-F]+\b',
            'ERROR_CODE': r'\b(?:E|ERR|ERROR)[_-]?[0-9]+\b|\b[A-Z]{2,}[0-9]{3,}\b',
            'SESSION_ID': r'\b[a-zA-Z0-9]{16,64}\b',
            'PID': r'\bpid[:\s]*[0-9]+\b',
            'THREAD_ID': r'\btid[:\s]*[0-9]+\b',
            
            # Time and numeric values
            'TIMESTAMP': r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b',
            'TIME_DURATION': r'\b\d+(?:\.\d+)?(?:ms|s|sec|min|h|hr|hour)s?\b',
            'MEMORY_SIZE': r'\b\d+(?:\.\d+)?(?:B|KB|MB|GB|TB)\b',
            'PERCENTAGE': r'\b\d+(?:\.\d+)?%\b',
            'NUMERIC_ID': r'\b[0-9]{6,}\b',
            
            # Email and user identifiers
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'USERNAME': r'\buser[:\s]*[a-zA-Z0-9._-]+\b',
            
            # Log-specific patterns
            'LOG_LEVEL': r'\b(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b',
            'HTTP_STATUS': r'\b[1-5][0-9]{2}\b',
            'DATABASE_QUERY': r'\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP)\b',
        }
    
    def _create_log_tokenizer(self) -> RegexpTokenizer:
        """
        Create a custom tokenizer for log messages.
        
        WHY CUSTOM TOKENIZATION FOR LOGS:
        - Standard word tokenizers break on punctuation, losing structure
        - Log messages contain meaningful punctuation (paths, IPs, etc.)
        - Need to preserve word boundaries while keeping structured tokens intact
        """
        
        # Pattern that captures:
        # 1. Words (including those with internal punctuation like file.txt)
        # 2. Numbers (including decimals)
        # 3. Special placeholders like <TOKEN_TYPE>
        tokenizer_pattern = r'\b\w+(?:[.-]\w+)*\b|\b\d+(?:\.\d+)?\b|<[A-Z_]+>|[^\w\s]'
        
        return RegexpTokenizer(tokenizer_pattern, gaps=False)
    
    def extract_and_normalize_special_tokens(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        Extract special tokens and replace them with normalized placeholders.
        
        Args:
            text: Input log message
            
        Returns:
            Tuple of (processed_text, extracted_tokens_dict)
        """
        
        extracted_tokens = defaultdict(list)
        processed_text = text
        
        if not self.preserve_special_tokens:
            return processed_text, dict(extracted_tokens)
        
        # Process each special pattern
        for token_type, pattern in self.special_patterns.items():
            matches = re.finditer(pattern, processed_text, re.IGNORECASE)
            
            for match in matches:
                token_value = match.group()
                extracted_tokens[token_type].append(token_value)
                
                # Replace with normalized placeholder
                placeholder = f"<{token_type}>"
                processed_text = processed_text.replace(token_value, placeholder, 1)
        
        return processed_text, dict(extracted_tokens)
    
    def basic_tokenize(self, text: str) -> List[str]:
        """
        Perform basic tokenization on preprocessed text.
        
        Args:
            text: Preprocessed log message
            
        Returns:
            List of tokens
        """
        
        # Extract placeholders first to preserve their case
        placeholder_map = {}
        placeholder_pattern = r'<[A-Z_]+>'
        
        # Find all placeholders and replace with temporary markers
        temp_text = text
        placeholder_count = 0
        for match in re.finditer(placeholder_pattern, text):
            placeholder = match.group()
            temp_marker = f"PLACEHOLDER{placeholder_count}"
            placeholder_map[temp_marker.lower()] = placeholder
            temp_text = temp_text.replace(placeholder, temp_marker, 1)
            placeholder_count += 1
        
        # Use custom log tokenizer on lowercase text
        tokens = self.log_tokenizer.tokenize(temp_text.lower())
        
        # Filter tokens and restore placeholders
        filtered_tokens = []
        for token in tokens:
            # Check if this token contains a placeholder marker
            restored_token = token
            for marker, placeholder in placeholder_map.items():
                if marker in token:
                    restored_token = token.replace(marker, placeholder)
                    break
            
            # Apply filtering only to non-placeholder tokens
            if restored_token.startswith('<') and restored_token.endswith('>'):
                filtered_tokens.append(restored_token)
            elif (len(restored_token) >= self.min_token_length and 
                  not restored_token.isspace() and 
                  not all(c in string.punctuation for c in restored_token)):
                filtered_tokens.append(restored_token)
        
        return filtered_tokens
    
    def remove_stopwords_func(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        WHY STOPWORDS REMOVAL IN LOGS:
        - Common words like 'the', 'and', 'is' rarely carry meaning in logs
        - Reduces noise for ML models
        - BUT: Keep domain-specific words that might seem like stopwords
        """
        
        if not self.remove_stopwords:
            return tokens
        
        # Only remove very common English stopwords that are truly noise
        minimal_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Use minimal stopwords instead of comprehensive list
        if self.remove_stopwords:
            all_stopwords = minimal_stopwords
        else:
            all_stopwords = set()
        
        filtered_tokens = []
        for token in tokens:
            # Always keep placeholder tokens
            if token.startswith('<') and token.endswith('>'):
                filtered_tokens.append(token)
            elif token.lower() not in all_stopwords:
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert treebank POS tag to WordNet POS tag for lemmatization."""
        
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun
    
    def apply_stemming_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming or lemmatization to tokens.
        
        STEMMING vs LEMMATIZATION FOR LOGS:
        
        STEMMING (NOT RECOMMENDED):
        - Fast but aggressive: "failed" -> "fail", "connection" -> "connect"
        - Can lose important distinctions: "processing" vs "process"
        - May create non-words: "database" -> "databas"
        
        LEMMATIZATION (RECOMMENDED):
        - Slower but accurate: "failed" -> "fail", "connections" -> "connection"
        - Preserves word meaning and readability
        - Better for log analysis where precision matters
        """
        
        if not (self.use_stemming or self.use_lemmatization):
            return tokens
        
        processed_tokens = []
        
        # Get POS tags for better lemmatization
        if self.use_lemmatization:
            try:
                pos_tags = pos_tag(tokens)
            except:
                pos_tags = [(token, 'NN') for token in tokens]  # Default to noun
        
        for i, token in enumerate(tokens):
            # Skip placeholder tokens
            if token.startswith('<') and token.endswith('>'):
                processed_tokens.append(token)
                continue
            
            # Skip if token is all digits or special characters
            if token.isdigit() or all(c in string.punctuation for c in token):
                processed_tokens.append(token)
                continue
            
            if self.use_stemming:
                try:
                    token = self.stemmer.stem(token)
                except:
                    pass  # Keep original if stemming fails
            
            if self.use_lemmatization:
                try:
                    # Get POS tag for this token
                    pos = self.get_wordnet_pos(pos_tags[i][1])
                    token = self.lemmatizer.lemmatize(token, pos=pos)
                except:
                    try:
                        # Fallback to default lemmatization
                        token = self.lemmatizer.lemmatize(token)
                    except:
                        pass  # Keep original if lemmatization fails
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    def tokenize_log_message(self, log_message: str) -> Dict[str, any]:
        """
        Complete tokenization pipeline for a single log message.
        
        Args:
            log_message: Raw log message
            
        Returns:
            Dictionary containing tokenization results
        """
        
        # Step 1: Extract and normalize special tokens
        processed_text, special_tokens = self.extract_and_normalize_special_tokens(log_message)
        
        # Step 2: Basic tokenization
        tokens = self.basic_tokenize(processed_text)
        
        # Step 3: Remove stopwords
        tokens = self.remove_stopwords_func(tokens)
        
        # Step 4: Apply stemming/lemmatization
        normalized_tokens = self.apply_stemming_lemmatization(tokens)
        
        # Step 5: Calculate statistics
        token_stats = {
            'total_tokens': len(normalized_tokens),
            'unique_tokens': len(set(normalized_tokens)),
            'special_token_types': len(special_tokens),
            'avg_token_length': np.mean([len(token) for token in normalized_tokens]) if normalized_tokens else 0
        }
        
        return {
            'original_text': log_message,
            'processed_text': processed_text,
            'tokens': normalized_tokens,
            'special_tokens': special_tokens,
            'token_stats': token_stats,
            'tokenized_string': ' '.join(normalized_tokens)
        }
    
    def tokenize_dataframe(self, df: pd.DataFrame, text_column: str = 'log_message') -> pd.DataFrame:
        """
        Apply tokenization pipeline to a DataFrame of log messages.
        
        Args:
            df: DataFrame containing log messages
            text_column: Name of column containing log text
            
        Returns:
            Enhanced DataFrame with tokenization results
        """
        
        print(f"🔤 Tokenizing {len(df)} log messages...")
        
        # Apply tokenization to each message
        tokenization_results = df[text_column].apply(self.tokenize_log_message)
        
        # Extract results into separate columns
        result_df = df.copy()
        result_df['tokens'] = tokenization_results.apply(lambda x: x['tokens'])
        result_df['tokenized_string'] = tokenization_results.apply(lambda x: x['tokenized_string'])
        result_df['special_tokens'] = tokenization_results.apply(lambda x: x['special_tokens'])
        result_df['token_count'] = tokenization_results.apply(lambda x: x['token_stats']['total_tokens'])
        result_df['unique_token_count'] = tokenization_results.apply(lambda x: x['token_stats']['unique_tokens'])
        result_df['special_token_types'] = tokenization_results.apply(lambda x: x['token_stats']['special_token_types'])
        
        print(f"✅ Tokenization complete!")
        print(f"📊 Average tokens per message: {result_df['token_count'].mean():.1f}")
        print(f"📊 Total unique tokens: {len(set([token for tokens in result_df['tokens'] for token in tokens]))}")
        
        return result_df
    
    def get_vocabulary_statistics(self, tokenized_df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate comprehensive vocabulary statistics from tokenized data.
        """
        
        # Collect all tokens
        all_tokens = []
        for token_list in tokenized_df['tokens']:
            all_tokens.extend(token_list)
        
        # Count token frequencies
        token_freq = defaultdict(int)
        for token in all_tokens:
            token_freq[token] += 1
        
        # Special token analysis
        special_token_counts = defaultdict(int)
        for special_dict in tokenized_df['special_tokens']:
            for token_type, tokens in special_dict.items():
                special_token_counts[token_type] += len(tokens)
        
        return {
            'total_tokens': len(all_tokens),
            'unique_tokens': len(set(all_tokens)),
            'vocabulary_size': len(token_freq),
            'avg_token_frequency': np.mean(list(token_freq.values())),
            'most_common_tokens': sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:20],
            'special_token_distribution': dict(special_token_counts),
            'token_length_distribution': {
                'mean': np.mean([len(token) for token in all_tokens]),
                'std': np.std([len(token) for token in all_tokens]),
                'min': min([len(token) for token in all_tokens]) if all_tokens else 0,
                'max': max([len(token) for token in all_tokens]) if all_tokens else 0
            }
        }


def demonstrate_log_tokenization():
    """
    Demonstrate the log tokenization pipeline with various examples.
    """
    
    print("=" * 80)
    print("LOG TOKENIZATION PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    # Sample log messages with various patterns
    sample_logs = [
        "2024-08-11 12:22:59 INFO User admin@company.com logged in from 192.168.1.100:8080",
        "2024-08-11 12:23:01 DEBUG Processing HTTP request to /api/v1/users.json with session abc123def456",
        "2024-08-11 12:23:03 ERROR Database connection failed: timeout after 30s connecting to db.internal:5432",
        "2024-08-11 12:23:05 WARN SSL certificate expired for https://api.service.com, error code E5001",
        "2024-08-11 12:23:07 CRITICAL Memory allocation failed at 0x7fff5fbff7e0, PID 12345 killed by OOM",
        "2024-08-11 12:23:09 INFO Backup completed: 1.2GB archived to /backup/data/2024-08-11.tar.gz"
    ]
    
    # Test different tokenization strategies
    configurations = [
        {
            'name': 'Basic Tokenization',
            'params': {
                'preserve_special_tokens': False,
                'use_stemming': False,
                'use_lemmatization': False,
                'remove_stopwords': False
            }
        },
        {
            'name': 'Log-Optimized (Recommended)',
            'params': {
                'preserve_special_tokens': True,
                'use_stemming': False,
                'use_lemmatization': True,
                'remove_stopwords': True
            }
        },
        {
            'name': 'Aggressive Stemming',
            'params': {
                'preserve_special_tokens': True,
                'use_stemming': True,
                'use_lemmatization': False,
                'remove_stopwords': True
            }
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame({'log_message': sample_logs})
    
    for config in configurations:
        print(f"\n{'='*50}")
        print(f"CONFIGURATION: {config['name']}")
        print(f"{'='*50}")
        
        # Initialize tokenizer
        tokenizer = LogTokenizer(**config['params'])
        
        # Process one example in detail
        example_log = sample_logs[2]  # Error message
        result = tokenizer.tokenize_log_message(example_log)
        
        print(f"\n📝 Example: {example_log}")
        print(f"🔤 Tokens: {result['tokens']}")
        print(f"🏷️  Special tokens: {result['special_tokens']}")
        print(f"📊 Token stats: {result['token_stats']}")
        
        # Process full dataset
        tokenized_df = tokenizer.tokenize_dataframe(df)
        vocab_stats = tokenizer.get_vocabulary_statistics(tokenized_df)
        
        print(f"\n📈 Vocabulary Statistics:")
        print(f"   Total tokens: {vocab_stats['total_tokens']}")
        print(f"   Unique tokens: {vocab_stats['unique_tokens']}")
        print(f"   Most common: {vocab_stats['most_common_tokens'][:5]}")
        print(f"   Special tokens: {vocab_stats['special_token_distribution']}")


if __name__ == "__main__":
    demonstrate_log_tokenization() 