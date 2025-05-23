import os
import time
import yaml
import logging
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import re
import json
import requests

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SnortLogHandler(FileSystemEventHandler):
    def __init__(self, config):
        self.config = config
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        
    def _load_model(self):
        """Load the trained model from the specified path"""
        try:
            if self.config['model_type'] == 'lstm':
                return tf.keras.models.load_model(self.config['model_path'])
            elif self.config['model_type'] == 'random_forest':
                import joblib
                return joblib.load(self.config['model_path'])
            else:
                raise ValueError(f"Unsupported model type: {self.config['model_type']}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def _load_scaler(self):
        """Load the feature scaler"""
        try:
            import joblib
            scaler_path = os.path.join(os.path.dirname(self.config['model_path']), 'scaler.pkl')
            return joblib.load(scaler_path)
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            return StandardScaler()
        
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.log'):
            logger.info(f"New log file detected: {event.src_path}")
            self._process_log_file(event.src_path)
            
    def _process_log_file(self, file_path):
        """Process a new Snort log file"""
        try:
            # Read and parse the log file
            raw_data = self._parse_snort_log(file_path)
            
            # Preprocess the data
            processed_data = self._preprocess_data(raw_data)
            
            # Make predictions
            predictions = self._make_predictions(processed_data)
            
            # Generate alerts if necessary
            self._generate_alerts(predictions, processed_data)
            
        except Exception as e:
            logger.error(f"Error processing log file {file_path}: {str(e)}")
            
    def _parse_snort_log(self, file_path):
        """Parse Snort log file into structured format"""
        log_entries = []
        
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    # Parse Snort log format
                    # Example: 01/15-14:20:54.647847  [**] [1:2008577:4] ET POLICY SSH Brute Force Attempt [**] [Classification: Attempted Administrator Privilege Gain] [Priority: 2] {TCP} 192.168.1.100:22 -> 10.0.0.5:49234
                    match = re.match(r'(\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+)\s+\[\*\*\]\s+\[(\d+:\d+:\d+)\]\s+(.*?)\s+\[\*\*\]\s+\[Classification:\s+(.*?)\]\s+\[Priority:\s+(\d+)\]\s+\{(\w+)\}\s+(\d+\.\d+\.\d+\.\d+):(\d+)\s+->\s+(\d+\.\d+\.\d+\.\d+):(\d+)', line)
                    
                    if match:
                        entry = {
                            'timestamp': match.group(1),
                            'signature_id': match.group(2),
                            'alert_message': match.group(3),
                            'classification': match.group(4),
                            'priority': int(match.group(5)),
                            'protocol': match.group(6),
                            'source_ip': match.group(7),
                            'source_port': int(match.group(8)),
                            'destination_ip': match.group(9),
                            'destination_port': int(match.group(10))
                        }
                        log_entries.append(entry)
                except Exception as e:
                    logger.warning(f"Error parsing log line: {str(e)}")
                    continue
                    
        return pd.DataFrame(log_entries)
        
    def _preprocess_data(self, raw_data):
        """Preprocess the parsed log data"""
        if raw_data.empty:
            return None
            
        # Convert timestamp to datetime
        raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], format='%m/%d-%H:%M:%S.%f')
        
        # Extract features specified in config
        features_df = pd.DataFrame()
        
        for feature in self.config['features']:
            if feature in raw_data.columns:
                features_df[feature] = raw_data[feature]
                
        # Handle categorical features
        features_df = pd.get_dummies(features_df, columns=['protocol'])
        
        # Normalize numerical features
        numerical_features = features_df.select_dtypes(include=[np.number]).columns
        features_df[numerical_features] = self.scaler.fit_transform(features_df[numerical_features])
        
        return features_df
        
    def _make_predictions(self, processed_data):
        """Run the model on preprocessed data"""
        if processed_data is None or processed_data.empty:
            return None
            
        try:
            if self.config['model_type'] == 'lstm':
                # Reshape data for LSTM
                X = processed_data.values.reshape((processed_data.shape[0], 1, processed_data.shape[1]))
                predictions = self.model.predict(X)
            else:
                predictions = self.model.predict(processed_data)
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
        
    def _generate_alerts(self, predictions, data):
        """Generate alerts based on model predictions"""
        if predictions is None or data is None:
            return
            
        try:
            # Apply threshold for anomaly detection
            threshold = self.config['threshold']
            alerts = []
            
            for i, (pred, row) in enumerate(zip(predictions, data.itertuples())):
                if pred > threshold:
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'severity': 'high' if pred > 0.95 else 'medium',
                        'source_ip': row.source_ip,
                        'destination_ip': row.destination_ip,
                        'confidence': float(pred),
                        'alert_type': 'Potential Security Threat',
                        'details': f"Anomalous traffic detected from {row.source_ip} to {row.destination_ip}"
                    }
                    alerts.append(alert)
                    
                    # Log alert
                    logger.warning(f"Alert generated: {json.dumps(alert)}")
                    
                    # Write to alert log file
                    with open(self.config['alerts']['log_file'], 'a') as f:
                        json.dump(alert, f)
                        f.write('\n')
                    
                    # Send notifications if configured
                    self._send_notifications(alert)
                    
        except Exception as e:
            logger.error(f"Error generating alerts: {str(e)}")
            
    def _send_notifications(self, alert):
        """Send alert notifications through configured channels"""
        try:
            if self.config['alerts']['notification']['webhook']:
                # Send webhook notification
                webhook_url = self.config.get('webhook_url')
                if webhook_url:
                    requests.post(webhook_url, json=alert)
                    
            if self.config['alerts']['notification']['syslog']:
                # Log to syslog
                logger.info(f"ALERT: {json.dumps(alert)}")
                
        except Exception as e:
            logger.error(f"Error sending notifications: {str(e)}")

def main():
    # Load configuration
    with open('configs/pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Create event handler and observer
    event_handler = SnortLogHandler(config)
    observer = Observer()
    
    # Set up monitoring for the Snort log directory
    observer.schedule(event_handler, config['snort_log_path'], recursive=False)
    observer.start()
    
    logger.info(f"Started monitoring Snort logs in {config['snort_log_path']}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopping Snort log monitoring")
    
    observer.join()

if __name__ == "__main__":
    main() 