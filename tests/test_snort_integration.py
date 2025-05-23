import unittest
import os
import tempfile
import yaml
from unittest.mock import Mock, patch
from pipeline.snort_integration import SnortLogHandler

class TestSnortIntegration(unittest.TestCase):
    def setUp(self):
        # Create a temporary config
        self.test_config = {
            'snort_log_path': '/tmp/snort/logs',
            'model_path': '/tmp/model.h5',
            'threshold': 0.85
        }
        
    def test_log_handler_initialization(self):
        handler = SnortLogHandler(self.test_config)
        self.assertEqual(handler.config, self.test_config)
        
    @patch('pipeline.snort_integration.SnortLogHandler._parse_snort_log')
    @patch('pipeline.snort_integration.SnortLogHandler._preprocess_data')
    @patch('pipeline.snort_integration.SnortLogHandler._make_predictions')
    @patch('pipeline.snort_integration.SnortLogHandler._generate_alerts')
    def test_process_log_file(self, mock_alerts, mock_predict, mock_preprocess, mock_parse):
        # Setup mock returns
        mock_parse.return_value = {'data': 'raw'}
        mock_preprocess.return_value = {'data': 'processed'}
        mock_predict.return_value = [1, 0, 1]
        
        handler = SnortLogHandler(self.test_config)
        handler._process_log_file('/tmp/test.log')
        
        # Verify all steps were called
        mock_parse.assert_called_once_with('/tmp/test.log')
        mock_preprocess.assert_called_once_with({'data': 'raw'})
        mock_predict.assert_called_once_with({'data': 'processed'})
        mock_alerts.assert_called_once_with([1, 0, 1], {'data': 'processed'})
        
    def test_invalid_log_file(self):
        handler = SnortLogHandler(self.test_config)
        with self.assertRaises(FileNotFoundError):
            handler._process_log_file('/nonexistent/file.log')

if __name__ == '__main__':
    unittest.main() 