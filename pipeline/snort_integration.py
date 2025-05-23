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
import joblib
import re
import json
import requests
from collections import defaultdict

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

class FlowAggregator:
    """Aggregates network flows and computes statistics"""
    def __init__(self, timeout=120):
        self.flows = defaultdict(list)
        self.timeout = timeout
        
    def add_packet(self, flow_key, packet_data):
        """Add a packet to its flow"""
        self.flows[flow_key].append(packet_data)
        
    def get_flow_features(self, flow_key):
        """Compute features for a flow"""
        packets = self.flows[flow_key]
        if not packets:
            return None
            
        # Basic flow statistics
        flow_duration = packets[-1]['timestamp'] - packets[0]['timestamp']
        total_bytes = sum(p.get('length', 0) for p in packets)
        total_packets = len(packets)
        
        # Separate forward and backward packets
        fwd_packets = [p for p in packets if p['direction'] == 'forward']
        bwd_packets = [p for p in packets if p['direction'] == 'backward']
        
        features = {
            'flow_duration': flow_duration,
            'flow_bytes_s': total_bytes / flow_duration if flow_duration > 0 else 0,
            'flow_packets_s': total_packets / flow_duration if flow_duration > 0 else 0,
            'total_fwd_packets': len(fwd_packets),
            'total_bwd_packets': len(bwd_packets),
            'total_length_fwd_packets': sum(p.get('length', 0) for p in fwd_packets),
            'total_length_bwd_packets': sum(p.get('length', 0) for p in bwd_packets)
        }
        
        # Add packet length statistics
        if fwd_packets:
            fwd_lengths = [p.get('length', 0) for p in fwd_packets]
            features.update({
                'fwd_packet_length_max': max(fwd_lengths),
                'fwd_packet_length_min': min(fwd_lengths),
                'fwd_packet_length_mean': np.mean(fwd_lengths)
            })
        
        if bwd_packets:
            bwd_lengths = [p.get('length', 0) for p in bwd_packets]
            features.update({
                'bwd_packet_length_max': max(bwd_lengths),
                'bwd_packet_length_min': min(bwd_lengths),
                'bwd_packet_length_mean': np.mean(bwd_lengths)
            })
            
        # Compute IAT (Inter Arrival Time) features
        packet_times = [p['timestamp'] for p in packets]
        iats = np.diff(packet_times)
        if len(iats) > 0:
            features.update({
                'flow_iat_mean': np.mean(iats),
                'flow_iat_std': np.std(iats),
                'flow_iat_max': np.max(iats),
                'flow_iat_min': np.min(iats)
            })
            
        # Forward and backward IAT means
        if len(fwd_packets) > 1:
            fwd_iats = np.diff([p['timestamp'] for p in fwd_packets])
            features['fwd_iat_mean'] = np.mean(fwd_iats)
            
        if len(bwd_packets) > 1:
            bwd_iats = np.diff([p['timestamp'] for p in bwd_packets])
            features['bwd_iat_mean'] = np.mean(bwd_iats)
            
        # Active and idle time
        if len(packets) > 1:
            active_times = []
            idle_times = []
            current_active = []
            
            for i in range(1, len(packet_times)):
                diff = packet_times[i] - packet_times[i-1]
                if diff > 2.0:  # 2 second threshold for idle time
                    if current_active:
                        active_times.append(sum(current_active))
                    idle_times.append(diff)
                    current_active = []
                else:
                    current_active.append(diff)
                    
            if active_times:
                features['active_mean'] = np.mean(active_times)
            if idle_times:
                features['idle_mean'] = np.mean(idle_times)
                
        return features

class ModelEnsemble:
    """Ensemble of LSTM and Random Forest models"""
    def __init__(self, config):
        self.config = config
        self.lstm_model = self._load_lstm()
        self.rf_model = self._load_random_forest()
        self.lstm_scaler = self._load_scaler('lstm')
        self.rf_scaler = self._load_scaler('random_forest')
        
    def _load_lstm(self):
        """Load LSTM model"""
        try:
            return tf.keras.models.load_model(self.config['models']['lstm']['path'])
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            return None
            
    def _load_random_forest(self):
        """Load Random Forest model"""
        try:
            return joblib.load(self.config['models']['random_forest']['path'])
        except Exception as e:
            logger.error(f"Error loading Random Forest model: {str(e)}")
            return None
            
    def _load_scaler(self, model_type):
        """Load scaler for specified model"""
        try:
            scaler_path = os.path.join(
                os.path.dirname(self.config['models'][model_type]['path']),
                'scaler.pkl'
            )
            return joblib.load(scaler_path)
        except Exception as e:
            logger.error(f"Error loading {model_type} scaler: {str(e)}")
            return StandardScaler()
            
    def predict(self, data):
        """Make ensemble prediction"""
        predictions = []
        weights = []
        
        # Get LSTM prediction if available
        if self.lstm_model is not None:
            lstm_features = self._prepare_lstm_features(data)
            if lstm_features is not None:
                lstm_pred = self.lstm_model.predict(lstm_features)
                predictions.append(lstm_pred)
                weights.append(self.config['models']['lstm']['weight'])
                
        # Get Random Forest prediction if available
        if self.rf_model is not None:
            rf_features = self._prepare_rf_features(data)
            if rf_features is not None:
                rf_pred = self.rf_model.predict_proba(rf_features)[:, 1]
                predictions.append(rf_pred)
                weights.append(self.config['models']['random_forest']['weight'])
                
        if not predictions:
            return None
            
        # Combine predictions based on ensemble method
        method = self.config['ensemble']['method']
        if method == 'weighted_average':
            weights = np.array(weights) / sum(weights)
            return np.average(predictions, axis=0, weights=weights)
        elif method == 'maximum':
            return np.maximum.reduce(predictions)
        elif method == 'minimum':
            return np.minimum.reduce(predictions)
        else:
            return np.mean(predictions, axis=0)
            
    def _prepare_lstm_features(self, data):
        """Prepare features for LSTM model"""
        try:
            sequence_features = self.config['features']['sequence']['features']
            sequence_length = self.config['features']['sequence']['sequence_length']
            
            # Select and scale features
            X = data[sequence_features].fillna(0)
            X_scaled = self.lstm_scaler.transform(X)
            
            # Reshape for LSTM (samples, sequence_length, features)
            n_samples = X_scaled.shape[0] - sequence_length + 1
            n_features = len(sequence_features)
            
            sequences = np.zeros((n_samples, sequence_length, n_features))
            for i in range(n_samples):
                sequences[i] = X_scaled[i:i + sequence_length]
                
            return sequences
            
        except Exception as e:
            logger.error(f"Error preparing LSTM features: {str(e)}")
            return None
            
    def _prepare_rf_features(self, data):
        """Prepare features for Random Forest model"""
        try:
            rf_features = self.config['features']['structured']
            
            # Select and scale features
            X = data[rf_features].fillna(0)
            X_scaled = self.rf_scaler.transform(X)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error preparing Random Forest features: {str(e)}")
            return None

class SnortLogHandler(FileSystemEventHandler):
    def __init__(self, config):
        self.config = config
        self.model_ensemble = ModelEnsemble(config)
        self.flow_aggregator = FlowAggregator(
            timeout=config['preprocessing']['feature_extraction']['flow_timeout']
        )
        
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
            
            # Extract flows and compute features
            processed_data = self._extract_flow_features(raw_data)
            
            if processed_data is not None and not processed_data.empty:
                # Make predictions using ensemble
                predictions = self.model_ensemble.predict(processed_data)
                
                # Generate alerts if necessary
                if predictions is not None:
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
                    match = re.match(r'(\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+)\s+\[\*\*\]\s+\[(\d+:\d+:\d+)\]\s+(.*?)\s+\[\*\*\]\s+\[Classification:\s+(.*?)\]\s+\[Priority:\s+(\d+)\]\s+\{(\w+)\}\s+(\d+\.\d+\.\d+\.\d+):(\d+)\s+->\s+(\d+\.\d+\.\d+\.\d+):(\d+)', line)
                    
                    if match:
                        timestamp = datetime.strptime(match.group(1), '%m/%d-%H:%M:%S.%f').timestamp()
                        entry = {
                            'timestamp': timestamp,
                            'signature_id': match.group(2),
                            'alert_message': match.group(3),
                            'classification': match.group(4),
                            'priority': int(match.group(5)),
                            'protocol': match.group(6),
                            'source_ip': match.group(7),
                            'source_port': int(match.group(8)),
                            'destination_ip': match.group(9),
                            'destination_port': int(match.group(10)),
                            'length': len(line)  # Use line length as packet length approximation
                        }
                        log_entries.append(entry)
                except Exception as e:
                    logger.warning(f"Error parsing log line: {str(e)}")
                    continue
                    
        return pd.DataFrame(log_entries)
        
    def _extract_flow_features(self, raw_data):
        """Extract flow features from raw data"""
        if raw_data.empty:
            return None
            
        # Group packets into flows
        for _, row in raw_data.iterrows():
            flow_key = f"{row['source_ip']}:{row['source_port']}-{row['destination_ip']}:{row['destination_port']}-{row['protocol']}"
            packet_data = {
                'timestamp': row['timestamp'],
                'length': row['length'],
                'direction': 'forward'
            }
            self.flow_aggregator.add_packet(flow_key, packet_data)
            
        # Extract features for each flow
        flow_features = []
        for flow_key in self.flow_aggregator.flows:
            features = self.flow_aggregator.get_flow_features(flow_key)
            if features:
                # Add basic flow information
                src_ip, src_port, dst_ip, dst_port, proto = re.match(
                    r'([^:]+):(\d+)-([^:]+):(\d+)-(\w+)',
                    flow_key
                ).groups()
                features.update({
                    'source_ip': src_ip,
                    'destination_ip': dst_ip,
                    'source_port': int(src_port),
                    'destination_port': int(dst_port),
                    'protocol': proto
                })
                flow_features.append(features)
                
        if not flow_features:
            return None
            
        return pd.DataFrame(flow_features)
        
    def _generate_alerts(self, predictions, data):
        """Generate alerts based on ensemble predictions"""
        if predictions is None or data is None:
            return
            
        try:
            threshold = self.config['ensemble']['threshold']
            alert_rules = self.config['alerts'].get('alert_rules', {})
            
            for i, (pred, row) in enumerate(zip(predictions, data.iterrows())):
                # Get the actual row data
                row = row[1]
                
                # Check prediction threshold
                if pred > threshold:
                    # Determine severity based on prediction confidence
                    if pred > 0.9:
                        severity = 'critical'
                    elif pred > 0.8:
                        severity = 'high'
                    elif pred > 0.6:
                        severity = 'medium'
                    else:
                        severity = 'low'
                        
                    # Check additional alert rules
                    alert_type = 'Anomaly Detected'
                    details = []
                    
                    if row.get('flow_packets_s', 0) > alert_rules.get('ddos_threshold', float('inf')):
                        alert_type = 'Potential DDoS Attack'
                        severity = 'critical'
                        details.append(f"Packet rate: {row['flow_packets_s']:.2f} packets/s")
                        
                    if row.get('total_length_fwd_packets', 0) > alert_rules.get('data_exfiltration_size', float('inf')):
                        alert_type = 'Potential Data Exfiltration'
                        severity = 'high'
                        details.append(f"Data transfer: {row['total_length_fwd_packets']/1024:.2f} KB")
                        
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'severity': severity,
                        'alert_type': alert_type,
                        'source_ip': row['source_ip'],
                        'destination_ip': row['destination_ip'],
                        'confidence': float(pred),
                        'ensemble_details': {
                            'method': self.config['ensemble']['method'],
                            'threshold': threshold
                        },
                        'flow_features': {
                            'packets_per_second': row.get('flow_packets_s', 0),
                            'bytes_per_second': row.get('flow_bytes_s', 0),
                            'duration': row.get('flow_duration', 0)
                        },
                        'details': details
                    }
                    
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