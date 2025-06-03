#!/usr/bin/env python3
"""
Quick test of the tokenizer to debug issues
"""

from src.log_tokenizer import LogTokenizer

def test_basic_tokenization():
    print("🧪 Testing Basic Tokenization")
    print("=" * 40)
    
    # Simple tokenizer without special processing
    tokenizer = LogTokenizer(
        preserve_special_tokens=False,
        use_stemming=False,
        use_lemmatization=False,
        remove_stopwords=False,
        min_token_length=1
    )
    
    test_message = "User failed to connect to database"
    result = tokenizer.tokenize_log_message(test_message)
    
    print(f"Original: {test_message}")
    print(f"Tokens: {result['tokens']}")
    print(f"Count: {len(result['tokens'])}")
    print()

def test_with_special_tokens():
    print("🧪 Testing With Special Tokens")
    print("=" * 40)
    
    tokenizer = LogTokenizer(
        preserve_special_tokens=True,
        use_stemming=False,
        use_lemmatization=False,
        remove_stopwords=False,
        min_token_length=1
    )
    
    test_message = "User admin@company.com failed to connect to 192.168.1.100"
    result = tokenizer.tokenize_log_message(test_message)
    
    print(f"Original: {test_message}")
    print(f"Processed: {result['processed_text']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Special: {result['special_tokens']}")
    print()

def test_full_pipeline():
    print("🧪 Testing Full Pipeline")
    print("=" * 40)
    
    tokenizer = LogTokenizer(
        preserve_special_tokens=True,
        use_stemming=False,
        use_lemmatization=True,
        remove_stopwords=True,
        min_token_length=2
    )
    
    test_message = "ERROR Database connections are failing intermittently"
    result = tokenizer.tokenize_log_message(test_message)
    
    print(f"Original: {test_message}")
    print(f"Processed: {result['processed_text']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Special: {result['special_tokens']}")
    print()

if __name__ == "__main__":
    test_basic_tokenization()
    test_with_special_tokens()
    test_full_pipeline() 