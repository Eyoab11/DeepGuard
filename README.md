# DeepGuard - Deep Learning IDS with Snort Integration

DeepGuard is a deep learning-based intrusion detection system that integrates with Snort logs to provide real-time threat detection and alerting.

## Required Files for Integration

### Essential Files (Not in Repository)
The following files are required but not included in the repository due to size:

1. Pre-trained model file:
   ```
   src/models/lstm/saved_models/latest.h5
   ```

2. Pre-trained scaler file:
   ```
   src/models/lstm/saved_models/scaler.pkl
   ```

You can obtain these files by:
- Downloading them from our secure storage (contact maintainers)
- OR using your own trained model and scaler (must match the expected input format)

### Files Included in Repository
- `pipeline/snort_integration.py` - Main integration code
- `configs/pipeline_config.yaml` - Configuration file
- All other supporting Python files

**Note:** You do NOT need any of the training data, notebooks, or other git-ignored files to run the integration. The pipeline only requires the pre-trained model files mentioned above.

## System Requirements

- Python 3.8 or higher
- Snort IDS installed and configured
- Git (for version control)
- 4GB RAM minimum (8GB recommended)
- Sufficient disk space for logs and model files

## Project Structure

```
DeepGuard/
├── configs/                 # Configuration files
│   └── pipeline_config.yaml # Main pipeline configuration
├── data/                   # Training data and datasets
├── logs/                   # Application logs
├── notebooks/             # Jupyter notebooks for analysis
├── pipeline/              # Main pipeline code
│   └── snort_integration.py
├── src/                   # Source code
│   └── models/           # ML model implementations
│       └── lstm/        # LSTM model files
├── tests/                # Unit tests
└── requirements.txt      # Python dependencies
```

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DeepGuard.git
   cd DeepGuard
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure the pipeline:
   - Open `configs/pipeline_config.yaml`
   - Update `snort_log_path` to point to your Snort logs directory
   - Adjust other settings as needed (thresholds, features, etc.)

## Running the Pipeline

1. Ensure your virtual environment is activated:
   ```bash
   # Windows
   .\venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

2. Start the pipeline:
   ```bash
   python pipeline/snort_integration.py
   ```

The pipeline will:
- Monitor the specified Snort log directory for new alerts
- Process incoming logs in real-time
- Apply the ML model for threat detection
- Generate alerts based on the detection results
- Log all activities and alerts

## Configuration Options

### Pipeline Configuration (pipeline_config.yaml)

```yaml
# Snort Configuration
snort_log_path: "/path/to/snort/logs"  # Update this
log_pattern: "*.log"

# Model Configuration
model_path: "./src/models/lstm/saved_models/latest.h5"
model_type: "lstm"
threshold: 0.85  # Detection threshold

# Feature Engineering
features:
  - source_ip
  - destination_ip
  - source_port
  - destination_port
  - protocol
  - priority
  - timestamp

# Alert Configuration
alerts:
  log_file: "./logs/alerts.log"
  severity_levels:
    - critical
    - high
    - medium
    - low
```

## Monitoring and Logs

- Pipeline logs: `logs/pipeline.log`
  - Contains operational logs and system status
  - Useful for debugging and monitoring

- Alert logs: `logs/alerts.log`
  - Contains generated security alerts
  - JSON format for easy parsing

## Troubleshooting

1. **Pipeline won't start:**
   - Check if virtual environment is activated
   - Verify all dependencies are installed
   - Ensure config paths are correct for your system

2. **No alerts being generated:**
   - Check if Snort logs are being written to the configured path
   - Verify log file permissions
   - Check pipeline logs for errors

3. **Model loading errors:**
   - Ensure model files exist in the specified path
   - Verify TensorFlow version compatibility

## Security Considerations

1. **Log Security:**
   - Regularly rotate logs
   - Monitor log file sizes
   - Secure access to log directories

2. **Model Security:**
   - Keep model files in a secure location
   - Regularly update and retrain models
   - Monitor for model drift

3. **Access Control:**
   - Use appropriate file permissions
   - Limit access to configuration files
   - Secure API endpoints if enabled

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Your License Here]

## Contact

[Your Contact Information]

---
**Note:** Large files (datasets, trained models) are not included in the repository. Contact the maintainers for access to these files.

Getachew0557