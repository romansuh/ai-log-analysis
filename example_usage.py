#!/usr/bin/env python3
"""
Simple Example Usage of Log Filtration Pipeline

This script demonstrates basic usage of the LogFilterPipeline
for extracting structured data from log messages.
"""

from log_parsing_pipeline import LogParsingPipeline

def main():
    pipeline = LogParsingPipeline()
    
    # Read logs from file
    with open('logs.txt', 'r') as f:
        logs = f.read().strip()
    
    print("LOGS BEFORE PARSING:")
    print(logs)
    
    print("\nLOGS AFTER PARSING:")
    df = pipeline.parse_log_file('logs.txt')
    print(df[['parsed_timestamp', 'log_level', 'message']].to_string(index=False))

if __name__ == "__main__":
    main() 