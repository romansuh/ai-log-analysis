#!/usr/bin/env python3
"""
Simple BERT Pipeline Test - Focused on Preprocessing and Tokenization Results
"""

import pandas as pd
import torch
from src.bert_pipeline import BERTConfig, LogBERTPipeline

def test_preprocessing_pipeline():
    """Test the preprocessing and tokenization pipeline with sample logs."""
    
    print("🔧 BERT LOG PREPROCESSING & TOKENIZATION TEST")
    print("=" * 55)
    
    # Sample realistic log messages
    sample_logs = [
        "2024-12-24 14:30:22 INFO User john.doe@company.com successfully authenticated from IP 192.168.1.100",
        "2024-12-24 14:30:25 ERROR Database connection failed: timeout after 30s connecting to db.internal:5432", 
        "2024-12-24 14:30:28 WARNING High memory usage: JVM heap at 85% (3.2GB/3.8GB), PID 8472",
        "2024-12-24 14:30:31 CRITICAL Security breach detected: unauthorized access to /admin/panel",
        "2024-12-24 14:30:34 DEBUG GET /api/v2/users/profile completed in 245ms, response size: 2.1KB"
    ]
    
    print(f"\n📝 Input: {len(sample_logs)} sample log messages")
    print("-" * 30)
    for i, log in enumerate(sample_logs, 1):
        print(f"{i}. {log[:80]}...")
    
    # Initialize pipeline with optimal settings
    config = BERTConfig(
        model_name="distilbert-base-uncased",  # Faster than full BERT
        max_length=128,                        # Optimal for log messages
        batch_size=8
    )
    
    pipeline = LogBERTPipeline(config)
    
    # Fit and transform
    print(f"\n🔄 Processing...")
    pipeline.fit(sample_logs)
    result = pipeline.transform(sample_logs)
    
    # Show results
    print(f"\n✅ PREPROCESSING & TOKENIZATION RESULTS")
    print("-" * 40)
    print(f"📊 Processed {len(sample_logs)} logs successfully")
    print(f"🔢 Output tensor shape: {result['input_ids'].shape}")
    print(f"🎯 Max sequence length: {config.max_length}")
    print(f"📏 Actual token lengths: {result['attention_mask'].sum(dim=1).tolist()}")
    
    # Show sample tokenization
    print(f"\n🔍 SAMPLE TOKENIZATION (First log):")
    print("-" * 35)
    
    # Get tokenizer for decoding (bert_tokenizer is directly the AutoTokenizer)
    tokenizer = pipeline.bert_tokenizer
    
    # Decode first sample
    first_tokens = result['input_ids'][0]
    decoded_tokens = tokenizer.convert_ids_to_tokens(first_tokens)
    
    # Show meaningful tokens (skip padding)
    meaningful_tokens = [token for token in decoded_tokens if token != '[PAD]'][:15]  # First 15 tokens
    
    print(f"Original: {sample_logs[0][:80]}...")
    print(f"Tokens:   {' | '.join(meaningful_tokens)}")
    print(f"Token IDs: {first_tokens[:len(meaningful_tokens)].tolist()}")
    
    # Pipeline statistics
    stats = pipeline.statistics_
    print(f"\n📈 PIPELINE STATISTICS:")
    print("-" * 25)
    print(f"• Average text length: {stats['avg_text_length']:.1f} chars")
    print(f"• Average tokens: {stats['avg_token_length']:.1f}")
    print(f"• Max tokens used: {stats['max_token_length']}")
    print(f"• Tokens over limit: {stats['tokens_exceeding_limit']}")
    
    print(f"\n🎯 Ready for ML training/inference!")

if __name__ == "__main__":
    test_preprocessing_pipeline() 