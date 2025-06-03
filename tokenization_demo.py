"""
Log Tokenization Pipeline Demonstration
=======================================

This script demonstrates advanced tokenization techniques for system log messages,
comparing different approaches and explaining their trade-offs.
"""

import pandas as pd
from src.log_tokenizer import LogTokenizer


def compare_tokenization_approaches():
    """
    Compare different tokenization approaches with real log examples.
    """
    
    print("🔤 LOG TOKENIZATION COMPARISON DEMO")
    print("=" * 60)
    
    # Real-world log examples with various complexities
    sample_logs = [
        # Authentication logs
        "2024-12-24 14:30:22 INFO [auth-service] User john.doe@company.com successfully authenticated from IP 10.0.1.15:443",
        
        # Database errors
        "2024-12-24 14:30:25 ERROR [db-pool] Connection timeout: failed to connect to postgres://db.internal:5432/userdb after 30000ms",
        
        # API requests with performance metrics
        "2024-12-24 14:30:28 DEBUG [api-gateway] GET /api/v2/users/12345/profile.json completed in 245ms, response size: 2.1KB",
        
        # Security events
        "2024-12-24 14:30:31 CRITICAL [security] Suspicious activity detected: 15 failed login attempts from 192.168.100.50 for user admin",
        
        # System performance
        "2024-12-24 14:30:34 WARN [monitor] High memory usage: JVM heap at 85% (3.2GB/3.8GB), PID 8472 in container web-app-7f9d8b",
        
        # File operations
        "2024-12-24 14:30:37 INFO [backup] Successfully archived 147 files (512MB) to /backup/daily/2024-12-24_14-30.tar.gz",
    ]
    
    # Create test DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-12-24 14:30:22', periods=len(sample_logs), freq='3S'),
        'log_message': sample_logs
    })
    
    print(f"📊 Processing {len(sample_logs)} sample log messages...")
    print()
    
    # Configuration 1: Minimal processing (baseline)
    print("🔹 APPROACH 1: MINIMAL PROCESSING")
    print("-" * 40)
    tokenizer_basic = LogTokenizer(
        preserve_special_tokens=False,
        use_stemming=False,
        use_lemmatization=False,
        remove_stopwords=False,
        min_token_length=1
    )
    
    example_result = tokenizer_basic.tokenize_log_message(sample_logs[1])
    print(f"Original: {example_result['original_text']}")
    print(f"Tokens:   {example_result['tokens'][:15]}...")  # Show first 15 tokens
    print(f"Count:    {example_result['token_stats']['total_tokens']} tokens")
    print()
    
    # Configuration 2: Log-optimized (recommended)
    print("🔹 APPROACH 2: LOG-OPTIMIZED (RECOMMENDED)")
    print("-" * 40)
    tokenizer_optimized = LogTokenizer(
        preserve_special_tokens=True,
        use_stemming=False,
        use_lemmatization=True,
        remove_stopwords=True,
        min_token_length=2
    )
    
    example_result = tokenizer_optimized.tokenize_log_message(sample_logs[1])
    print(f"Original:      {example_result['original_text']}")
    print(f"Processed:     {example_result['processed_text']}")
    print(f"Tokens:        {example_result['tokens']}")
    print(f"Special tokens: {example_result['special_tokens']}")
    print(f"Count:         {example_result['token_stats']['total_tokens']} tokens")
    print()
    
    # Configuration 3: Aggressive processing
    print("🔹 APPROACH 3: AGGRESSIVE PROCESSING")
    print("-" * 40)
    tokenizer_aggressive = LogTokenizer(
        preserve_special_tokens=True,
        use_stemming=True,
        use_lemmatization=False,
        remove_stopwords=True,
        min_token_length=3
    )
    
    example_result = tokenizer_aggressive.tokenize_log_message(sample_logs[1])
    print(f"Original: {example_result['original_text']}")
    print(f"Tokens:   {example_result['tokens']}")
    print(f"Count:    {example_result['token_stats']['total_tokens']} tokens")
    print()
    
    # Process full dataset with recommended approach
    print("🔹 FULL DATASET ANALYSIS")
    print("-" * 40)
    
    tokenized_df = tokenizer_optimized.tokenize_dataframe(df)
    vocab_stats = tokenizer_optimized.get_vocabulary_statistics(tokenized_df)
    
    print(f"\n📈 Vocabulary Statistics:")
    print(f"   • Total tokens: {vocab_stats['total_tokens']}")
    print(f"   • Unique tokens: {vocab_stats['unique_tokens']}")
    print(f"   • Vocabulary size: {vocab_stats['vocabulary_size']}")
    print(f"   • Avg token length: {vocab_stats['token_length_distribution']['mean']:.1f} characters")
    
    print(f"\n🏆 Most Common Tokens:")
    for token, count in vocab_stats['most_common_tokens'][:10]:
        print(f"   • '{token}': {count} occurrences")
    
    print(f"\n🏷️  Special Token Distribution:")
    for token_type, count in vocab_stats['special_token_distribution'].items():
        print(f"   • {token_type}: {count} instances")
    
    return tokenized_df


def demonstrate_special_token_handling():
    """
    Demonstrate special token extraction and normalization.
    """
    
    print("\n" + "=" * 60)
    print("🏷️  SPECIAL TOKEN HANDLING DEMONSTRATION")
    print("=" * 60)
    
    # Log messages rich in special tokens
    special_examples = [
        "User session abc123def456789 from 192.168.1.100:8080 accessed https://api.company.com/v1/data.json",
        "Failed to connect to database server at db.internal:5432, error code DB_CONN_ERR_001",
        "Memory allocation failed at 0x7fff5fbff7e0, process PID 12345 terminated with exit code 137",
        "SSL certificate for domain *.company.com expires in 7 days, renewal required",
        "Backup job uuid: 550e8400-e29b-41d4-a716-446655440000 completed, archived 2.5GB to /backup/2024/12/daily.tar.gz",
        "HTTP 403 response for POST /api/v1/users from client 10.0.0.50 (Mozilla/5.0 Chrome/96.0)",
    ]
    
    tokenizer = LogTokenizer(preserve_special_tokens=True)
    
    for i, log in enumerate(special_examples, 1):
        print(f"\n📝 Example {i}:")
        print(f"Original: {log}")
        
        result = tokenizer.tokenize_log_message(log)
        
        print(f"Processed: {result['processed_text']}")
        print(f"Tokens: {result['tokens']}")
        
        if result['special_tokens']:
            print("🏷️  Extracted special tokens:")
            for token_type, tokens in result['special_tokens'].items():
                print(f"   • {token_type}: {tokens}")


def analyze_stemming_vs_lemmatization():
    """
    Compare stemming vs lemmatization for log analysis.
    """
    
    print("\n" + "=" * 60)
    print("⚖️  STEMMING vs LEMMATIZATION ANALYSIS")
    print("=" * 60)
    
    test_logs = [
        "Database connections are failing intermittently during peak processing times",
        "User authentication services have been experiencing timeouts and delays",
        "The backup processes completed successfully after running for several hours",
        "Multiple failed login attempts were detected from various IP addresses",
        "Server responses are slower than expected, causing client-side timeouts",
    ]
    
    # Stemming approach
    print("🔹 STEMMING APPROACH:")
    print("-" * 25)
    stemmer_tokenizer = LogTokenizer(
        preserve_special_tokens=False,
        use_stemming=True,
        use_lemmatization=False,
        remove_stopwords=True
    )
    
    stemmed_results = []
    for log in test_logs:
        result = stemmer_tokenizer.tokenize_log_message(log)
        stemmed_results.append(result['tokens'])
        print(f"Original: {log}")
        print(f"Stemmed:  {' '.join(result['tokens'])}")
        print()
    
    # Lemmatization approach
    print("🔹 LEMMATIZATION APPROACH:")
    print("-" * 30)
    lemma_tokenizer = LogTokenizer(
        preserve_special_tokens=False,
        use_stemming=False,
        use_lemmatization=True,
        remove_stopwords=True
    )
    
    lemma_results = []
    for log in test_logs:
        result = lemma_tokenizer.tokenize_log_message(log)
        lemma_results.append(result['tokens'])
        print(f"Original:    {log}")
        print(f"Lemmatized:  {' '.join(result['tokens'])}")
        print()
    
    # Analysis
    print("📊 COMPARISON ANALYSIS:")
    print("-" * 25)
    
    all_stemmed = [token for tokens in stemmed_results for token in tokens]
    all_lemmatized = [token for tokens in lemma_results for token in tokens]
    
    print(f"Stemming vocabulary size: {len(set(all_stemmed))}")
    print(f"Lemmatization vocabulary size: {len(set(all_lemmatized))}")
    
    print("\n🎯 RECOMMENDATION FOR LOG ANALYSIS:")
    print("   ✅ Use LEMMATIZATION because:")
    print("   • Preserves word readability and meaning")
    print("   • Maintains semantic relationships important for log analysis")
    print("   • Produces valid English words that are easier to interpret")
    print("   • Better for troubleshooting and pattern recognition")
    print("   • Slight vocabulary increase is acceptable for log analysis precision")


def regex_pattern_examples():
    """
    Demonstrate the regex patterns used for special token detection.
    """
    
    print("\n" + "=" * 60)
    print("🔍 REGEX PATTERN EXAMPLES")
    print("=" * 60)
    
    tokenizer = LogTokenizer()
    
    print("📋 DEFINED PATTERNS:")
    patterns_to_show = [
        ('IPV4_ADDRESS', '192.168.1.100, 10.0.0.1, 255.255.255.0'),
        ('EMAIL', 'user@company.com, admin@localhost.local'),
        ('UNIX_PATH', '/var/log/app.log, /home/user/documents/file.txt'),
        ('UUID', '550e8400-e29b-41d4-a716-446655440000'),
        ('HEX_VALUE', '0x7fff5fbff7e0, 0xDEADBEEF'),
        ('ERROR_CODE', 'ERR_001, ERROR-404, DB_CONN_ERR_001'),
        ('TIME_DURATION', '30s, 2.5min, 1.2hr, 500ms'),
        ('MEMORY_SIZE', '2.5GB, 512MB, 1024KB, 2048B'),
        ('HTTP_STATUS', '200, 404, 500, 301'),
    ]
    
    for pattern_name, examples in patterns_to_show:
        pattern = tokenizer.special_patterns[pattern_name]
        print(f"\n🔸 {pattern_name}:")
        print(f"   Pattern: {pattern}")
        print(f"   Examples: {examples}")
    
    print(f"\n📊 Total patterns defined: {len(tokenizer.special_patterns)}")


if __name__ == "__main__":
    # Run all demonstrations
    print("🚀 Starting Log Tokenization Pipeline Demonstration\n")
    
    # Main comparison
    tokenized_data = compare_tokenization_approaches()
    
    # Special token handling
    demonstrate_special_token_handling()
    
    # Stemming vs lemmatization
    analyze_stemming_vs_lemmatization()
    
    # Regex patterns
    regex_pattern_examples()
    
    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("📚 Key Takeaways:")
    print("   1. Use lemmatization over stemming for log analysis")
    print("   2. Preserve special tokens with normalization")
    print("   3. Remove stopwords but keep domain-specific terms")
    print("   4. Custom tokenization handles log structure better than standard tokenizers")
    print("   5. Regular expressions are crucial for extracting meaningful patterns") 