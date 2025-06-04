#!/usr/bin/env python3
"""
Simple Log Filtration Pipeline

A straightforward log parser that extracts structured data from raw log messages.
Separates metadata (timestamps, log levels, modules) from main message content.
"""

import re
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import warnings

class LogParsingPipeline:
    """
    A simple log parser that extracts structured information from raw log messages.
    
    Supports common log formats:
    - Simple format: "YYYY-MM-DD HH:MM:SS LEVEL MESSAGE"
    - Complex format: "YYYY-MM-DD HH:MM:SS,mmm - LEVEL [module/thread] - MESSAGE"
    """
    
    def __init__(self):
        """Initialize the log filter pipeline with predefined patterns."""
        self.patterns = {
            'simple': {
                'regex': r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+(.+)',
                'groups': ['timestamp', 'log_level', 'message'],
                'timestamp_format': '%Y-%m-%d %H:%M:%S'
            },
            'complex': {
                'regex': r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+-\s+(\w+)\s+\[([^\]]+)\]\s+-\s+(.+)',
                'groups': ['timestamp', 'log_level', 'module_thread', 'message'],
                'timestamp_format': '%Y-%m-%d %H:%M:%S,%f'
            }
        }
    
    def add_custom_pattern(self, name: str, regex: str, groups: List[str], 
                          timestamp_format: Optional[str] = None) -> None:
        """Add a custom log pattern for parsing."""
        self.patterns[name] = {
            'regex': regex,
            'groups': groups,
            'timestamp_format': timestamp_format
        }
    
    def _detect_pattern(self, log_line: str) -> Optional[str]:
        """Auto-detect which pattern matches the log line."""
        for pattern_name, pattern_config in self.patterns.items():
            if re.match(pattern_config['regex'], log_line.strip()):
                return pattern_name
        return None
    
    def _parse_timestamp(self, timestamp_str: str, format_str: str) -> Optional[datetime]:
        """Parse timestamp string into datetime object."""
        try:
            if ',%f' in format_str and ',' in timestamp_str:
                timestamp_str = timestamp_str.replace(',', '.')
                parts = timestamp_str.split('.')
                if len(parts) == 2 and len(parts[1]) == 3:
                    timestamp_str = f"{parts[0]}.{parts[1]}000"
                format_str = format_str.replace(',%f', '.%f')
            
            return datetime.strptime(timestamp_str, format_str)
        except ValueError:
            return None
    
    def parse_log_line(self, log_line: str, pattern: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse a single log line into structured data."""
        log_line = log_line.strip()
        if not log_line:
            return None
        
        if pattern is None:
            pattern = self._detect_pattern(log_line)
        
        if pattern is None or pattern not in self.patterns:
            return {'raw_message': log_line, 'message': log_line}
        
        pattern_config = self.patterns[pattern]
        match = re.match(pattern_config['regex'], log_line)
        
        if not match:
            return {'raw_message': log_line, 'message': log_line}
        
        parsed_data = {}
        groups = match.groups()
        field_names = pattern_config['groups']
        
        for i, field_name in enumerate(field_names):
            if i < len(groups):
                parsed_data[field_name] = groups[i]
        
        if 'timestamp' in parsed_data and pattern_config.get('timestamp_format'):
            parsed_timestamp = self._parse_timestamp(
                parsed_data['timestamp'], 
                pattern_config['timestamp_format']
            )
            if parsed_timestamp:
                parsed_data['parsed_timestamp'] = parsed_timestamp
        
        parsed_data['raw_message'] = log_line
        return parsed_data
    
    def parse_logs(self, log_data: Union[str, List[str]], pattern: Optional[str] = None) -> pd.DataFrame:
        """Parse multiple log lines and return a structured DataFrame."""
        if isinstance(log_data, str):
            log_lines = log_data.strip().split('\n')
        elif isinstance(log_data, list):
            log_lines = log_data
        else:
            raise ValueError("log_data must be either a string or list of strings")
        
        parsed_logs = []
        for log_line in log_lines:
            parsed = self.parse_log_line(log_line, pattern)
            if parsed:
                parsed_logs.append(parsed)
        
        if not parsed_logs:
            return pd.DataFrame()
        
        df = pd.DataFrame(parsed_logs)
        return self._organize_dataframe(df)
    
    def parse_log_file(self, file_path: str, pattern: Optional[str] = None, 
                      encoding: str = 'utf-8') -> pd.DataFrame:
        """Parse a log file and return structured DataFrame."""
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                log_data = file.read()
            return self.parse_logs(log_data, pattern)
        except Exception as e:
            warnings.warn(f"Error reading file {file_path}: {e}")
            return pd.DataFrame()
    
    def _organize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Organize and clean the DataFrame columns."""
        priority_cols = ['parsed_timestamp', 'timestamp', 'log_level', 'message', 'module_thread']
        
        existing_priority = [col for col in priority_cols if col in df.columns]
        other_cols = [col for col in df.columns if col not in priority_cols and col != 'raw_message']
        
        new_order = existing_priority + other_cols
        if 'raw_message' in df.columns:
            new_order.append('raw_message')
        
        df = df[new_order]
        
        if 'parsed_timestamp' in df.columns:
            df['parsed_timestamp'] = pd.to_datetime(df['parsed_timestamp'], errors='coerce')
        
        if 'log_level' in df.columns:
            df['log_level'] = df['log_level'].str.upper()
        
        if 'message' in df.columns:
            df['message'] = df['message'].str.lower()
        
        return df


if __name__ == "__main__":
    pipeline = LogParsingPipeline()
    
    # Show logs before parsing
    print("LOGS BEFORE PARSING:")
    with open('logs.txt', 'r') as f:
        raw_logs = f.read()
    print(raw_logs)
    
    print("\nLOGS AFTER PARSING:")
    df = pipeline.parse_log_file('logs.txt')
    print(df[['parsed_timestamp', 'log_level', 'message']].to_string(index=False)) 