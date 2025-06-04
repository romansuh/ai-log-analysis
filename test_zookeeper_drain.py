#!/usr/bin/env python3
"""
Test Drain templating with Zookeeper logs
"""

from log_parsing_pipeline import LogParsingPipeline

def main():
    pipeline = LogParsingPipeline()
    
    # Test with first 20 lines of Zookeeper logs
    df = pipeline.process_logs('Zookeeper_2k.log')
    
    # Show first 10 results
    print("ZOOKEEPER LOGS - DRAIN TEMPLATING TEST:")
    print("\nFIRST 10 PROCESSED LOGS:")
    subset = df.head(10)[['message', 'template', 'cluster_id']]
    
    for i, row in subset.iterrows():
        print(f"{i+1:2d}. Message: {row['message']}")
        print(f"    Template: {row['template']}")
        print(f"    Cluster:  {row['cluster_id']}")
        print()

if __name__ == "__main__":
    main() 