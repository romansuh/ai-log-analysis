#!/usr/bin/env python3
"""
Clean Log Parsing Pipeline with Drain
"""

import pandas as pd
import re
from drain3 import TemplateMiner

class LogParsingPipeline:
    """Simple log parser using Drain algorithm."""
    
    def __init__(self):
        self.template_miner = TemplateMiner()
    
    def _preprocess_message(self, message: str) -> str:
        """Preprocess message for better Drain pattern matching."""
        message = re.sub(r'\b\d+\b', '<NUM>', message)
        message = re.sub(r'\b\d+\.\d+\.\d+\.\d+\b', '<IP>', message)
        message = re.sub(r':\d+', ':<PORT>', message)
        return message
    
    def _parse_log_line(self, log_line: str) -> dict:
        """Parse a single log line into components."""
        patterns = [
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+(.+)',
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
            result.update({
                'timestamp': match.group(1),
                'log_level': match.group(2).upper(),
                'message': match.group(3).lower()
            })
            return result
        
        # Try complex format
        match = re.match(patterns[1], log_line)
        if match:
            result.update({
                'timestamp': match.group(1),
                'log_level': match.group(2).upper(),
                'module_thread': match.group(3),
                'message': match.group(4).lower()
            })
            return result
        
        # Fallback: treat whole line as message
        result['message'] = log_line.lower()
        return result
    
    def process_logs(self, file_path: str) -> pd.DataFrame:
        """Process logs from file using Drain algorithm."""
        with open(file_path, 'r') as f:
            raw_logs = [line.strip() for line in f if line.strip()]

        parsed_logs = []
        for log in raw_logs:
            parsed = self._parse_log_line(log)
            
            if parsed['message']:
                preprocessed = self._preprocess_message(parsed['message'])
                result = self.template_miner.add_log_message(preprocessed)
                parsed['template'] = result.get('template_mined', parsed['message'])
                parsed['Drain_cluster_id'] = result.get('cluster_id')
            else:
                parsed['template'] = None
                parsed['Drain_cluster_id'] = None
            
            parsed_logs.append(parsed)
        
        df = pd.DataFrame(parsed_logs)
        
        # Convert timestamp if possible
        if 'timestamp' in df.columns and df['timestamp'].notna().any():
            df['parsed_timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df

if __name__ == "__main__":
    pipeline = LogParsingPipeline()
    df = pipeline.process_logs('logs.txt')
    
    print(f"Processed {len(df)} log entries")
    print("\nParsed logs:")
    print(df[['timestamp', 'log_level', 'message', 'template']].to_string(index=False)) 