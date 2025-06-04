# Simple Log Filtration Pipeline

A straightforward Python tool for parsing log files and extracting structured data. Separates metadata (timestamps, log levels, module names) from the main log message content into clean pandas DataFrames.

## Features

- **Automatic Pattern Detection**: Recognizes common log formats automatically
- **Clean Data Extraction**: Separates timestamps, log levels, and messages into distinct columns
- **Flexible Input**: Parse strings, lists, or files
- **Custom Patterns**: Add your own regex patterns for specific log formats
- **Pandas Integration**: Returns structured DataFrames ready for analysis

## Quick Start

```python
from log_filtration_pipeline import LogFilterPipeline

# Initialize the pipeline
pipeline = LogFilterPipeline()

# Parse a log file
df = pipeline.parse_log_file('your_logs.txt')

# View structured data (metadata separated from message content)
print(df[['parsed_timestamp', 'log_level', 'message']])
```

## Supported Log Formats

### Simple Format
```
2024-08-11 12:22:59 INFO User logged in
2024-08-11 12:23:01 DEBUG Fetching user details
2024-08-11 12:23:03 ERROR Failed to connect to database
```

### Complex Format (with module/thread info)
```
2015-07-29 17:41:44,747 - INFO  [QuorumPeer[myid=1]:FastLeaderElection@774] - Notification time out: 3200
2015-07-29 19:04:12,394 - INFO  [/10.10.34.11:3888:QuorumCnxManager$Listener@493] - Received connection request
```

## Usage Examples

### Parse Individual Log Lines
```python
pipeline = LogFilterPipeline()
log_line = "2024-08-11 15:30:45 WARN Connection timeout detected"
parsed = pipeline.parse_log_line(log_line)

print(f"Message: {parsed['message']}")
print(f"Level: {parsed['log_level']}")
print(f"Timestamp: {parsed['timestamp']}")
```

### Parse Multiple Logs
```python
logs = """
2024-08-11 12:22:59 INFO User logged in
2024-08-11 12:23:01 DEBUG Fetching user details
2024-08-11 12:23:03 ERROR Failed to connect to database
"""

df = pipeline.parse_logs(logs)
print(df[['parsed_timestamp', 'log_level', 'message']])
```

### Filter Parsed Data
```python
# Get only error and warning messages
error_logs = df[df['log_level'].isin(['ERROR', 'WARN'])]

# Extract clean message content (no metadata)
clean_messages = df['message'].tolist()
```

### Add Custom Patterns
```python
# Add pattern for web server logs
web_pattern = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s+\[([^\]]+)\]\s+"(\w+)\s+([^"]+)"\s+(\d+)'
pipeline.add_custom_pattern(
    name='web_server',
    regex=web_pattern,
    groups=['client_ip', 'timestamp', 'method', 'url', 'status_code']
)

# Parse with custom pattern
web_log = '192.168.1.100 [29/Jul/2024:10:15:30] "GET /api/users" 200'
parsed = pipeline.parse_log_line(web_log, pattern='web_server')
```

## Output Structure

The pipeline outputs a pandas DataFrame with these columns (when available):

- `parsed_timestamp`: Parsed datetime object
- `timestamp`: Original timestamp string
- `log_level`: Log level (INFO, ERROR, WARN, DEBUG, etc.)
- `message`: Main log message content (cleaned of metadata)
- `module_thread`: Module or thread information (for complex formats)
- `raw_message`: Original unparsed log line

## Requirements

- pandas
- Python 3.7+

## Files

- `log_filtration_pipeline.py`: Main pipeline implementation
- `example_usage.py`: Usage examples and demonstrations
- `logs.txt`: Sample simple log file
- `Zookeeper_2k.log`: Sample complex log file

## Running Examples

```bash
# Run the main pipeline demo
python log_filtration_pipeline.py

# Run detailed usage examples
python example_usage.py
``` 