# DeepGuard

A deep learning-based intrusion detection system that integrates with Snort logs for real-time threat detection and analysis.

## Project Structure

```
DeepGuard/
├── data/                  # Data directory
├── src/                   # Source code
├── notebooks/            # Jupyter notebooks
├── configs/              # Configuration files
├── tests/               # Unit tests
├── logs/                # Log files
└── pipeline/            # Pipeline scripts
```

## Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/DeepGuard.git
cd DeepGuard
```

2. Download the datasets
```bash
# Create necessary directories
mkdir -p data/{raw,processed,external}

# Download datasets from releases
# Visit: https://github.com/yourusername/DeepGuard/releases
# Download the following files:
# - CICIDS2017.zip
# - UNSW.zip
# - Binary-Labeled-CICIDS.zip

# Extract to appropriate directories
unzip CICIDS2017.zip -d data/external/
unzip UNSW.zip -d data/external/
unzip Binary-Labeled-CICIDS.zip -d data/raw/
```

3. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

## Pipeline Integration

The system integrates with Snort logs through the following pipeline:
1. Log ingestion from Snort
2. Preprocessing and feature extraction
3. Model inference
4. Alert generation

## Usage

Detailed usage instructions and examples can be found in the documentation.

## Data

Due to file size limitations, the datasets are not stored directly in the repository. Please download them from the releases page:

- CICIDS2017 Dataset: Network traffic data with various attack types
- UNSW-NB15 Dataset: Network intrusion detection dataset
- Binary-Labeled-CICIDS: Preprocessed CICIDS2017 data for binary classification

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Getachew0557