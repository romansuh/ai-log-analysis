#!/usr/bin/env python3
"""
Simple Log Filtration Pipeline with Drain
"""

import pandas as pd
import re
from datetime import datetime
from drain3 import TemplateMiner

class LogParsingPipeline:
    """Simple log parser using Drain algorithm."""
    
    def __init__(self):
        """Initialize the pipeline with Drain."""
        self.template_miner = TemplateMiner()
    
    def _preprocess_message(self, message: str) -> str:
        """Preprocess message to help Drain find patterns better."""
        # Replace numbers with placeholders to help Drain create templates
        # Replace specific numbers, IDs, timestamps, IPs
        message = re.sub(r'\b\d+\b', '<NUM>', message)           # Numbers
        message = re.sub(r'\b\d+\.\d+\.\d+\.\d+\b', '<IP>', message)  # IP addresses
        message = re.sub(r':\d+', ':<PORT>', message)           # Ports
        
        return message
    
    def _parse_log_line(self, log_line: str) -> dict:
        """Parse a single log line into components."""
        # Patterns for different log formats
        patterns = [
            # Simple format: YYYY-MM-DD HH:MM:SS LEVEL MESSAGE
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+(.+)',
            # Complex format: YYYY-MM-DD HH:MM:SS,mmm - LEVEL [module] - MESSAGE  
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+-\s+(\w+)\s+\[([^\]]+)\]\s+-\s+(.+)'
        ]
        
        result = {
            'raw_log': log_line,
            'timestamp': None,
            'log_level': None,
            'message': None,
            'module_thread': None
        }
        
        # Try simple format first
        match = re.match(patterns[0], log_line)
        if match:
            result['timestamp'] = match.group(1)
            result['log_level'] = match.group(2).upper()
            result['message'] = match.group(3).lower()
            return result
        
        # Try complex format
        match = re.match(patterns[1], log_line)
        if match:
            result['timestamp'] = match.group(1)
            result['log_level'] = match.group(2).upper()
            result['module_thread'] = match.group(3)
            result['message'] = match.group(4).lower()
            return result
        
        # If no pattern matches, treat whole line as message
        result['message'] = log_line.lower()
        return result
    
    def process_logs(self, file_path: str) -> pd.DataFrame:
        """Process logs from file using Drain algorithm."""
        # 1. Load logs to DataFrame
        with open(file_path, 'r') as f:
            raw_logs = f.readlines()

        # 2. Process logs - parse each line into components
        parsed_logs = []
        for log in raw_logs:
            log = log.strip()
            if log:
                parsed = self._parse_log_line(log)
                
                # Use Drain on the message part only
                if parsed['message']:
                    # Preprocess message to help Drain find patterns
                    preprocessed = self._preprocess_message(parsed['message'])
                    result = self.template_miner.add_log_message(preprocessed)
                    parsed['template'] = result.get('template_mined', parsed['message'])
                    parsed['cluster_id'] = result.get('cluster_id')
                else:
                    parsed['template'] = None
                    parsed['cluster_id'] = None
                
                parsed_logs.append(parsed)
        
        df = pd.DataFrame(parsed_logs)
        
        # Convert timestamp to datetime if possible
        if 'timestamp' in df.columns and df['timestamp'].notna().any():
            df['parsed_timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df

if __name__ == "__main__":
    pipeline = LogParsingPipeline()
    
    # Process logs
    df = pipeline.process_logs('logs.txt')
    
    # 3. Display unprocessed and processed logs
    print("UNPROCESSED LOGS:")
    for log in df['raw_log']:
        print(log)
    
    print("\nPROCESSED LOGS (Broken down):")
    print(df[['timestamp', 'log_level', 'message', 'template']].to_string(index=False))
    
    print(f"\nDATAFRAME COLUMNS: {list(df.columns)}")
    print(f"DATAFRAME SHAPE: {df.shape}") 