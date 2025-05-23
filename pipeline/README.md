# Pipeline Directory

This directory contains the automation pipeline scripts for integrating Snort logs with the DeepGuard system.

## Components

1. `snort_ingestion.py`: Script for ingesting and parsing Snort logs
2. `feature_extraction.py`: Extracts relevant features from the logs
3. `model_inference.py`: Runs the trained model on processed data
4. `alert_generator.py`: Generates and manages security alerts

## Pipeline Flow

1. **Log Ingestion**
   - Monitors Snort log directory
   - Parses incoming logs in real-time
   - Validates log format and content

2. **Preprocessing**
   - Cleans and normalizes log data
   - Extracts relevant features
   - Prepares data for model inference

3. **Model Inference**
   - Runs preprocessed data through trained models
   - Generates prediction scores
   - Applies detection thresholds

4. **Alert Generation**
   - Creates detailed alerts for detected threats
   - Prioritizes alerts based on severity
   - Integrates with security monitoring systems

## Configuration

The pipeline can be configured through the `config.yaml` file in the `configs` directory. 