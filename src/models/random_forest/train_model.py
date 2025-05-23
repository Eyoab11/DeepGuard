import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_random_forest():
    """Train Random Forest model on CICIDS dataset"""
    print("Loading and preprocessing CICIDS dataset...")
    
    # Create directories if they don't exist
    os.makedirs('src/models/random_forest/saved_models', exist_ok=True)
    
    # Load the preprocessed CICIDS data
    # Note: These files should be provided separately due to size
    X_train = pd.read_csv('data/processed/cicids/X_train.csv')
    y_train = pd.read_csv('data/processed/cicids/y_train.csv')
    
    print(f"Training data shape: {X_train.shape}")
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save the scaler
    joblib.dump(scaler, 'src/models/random_forest/saved_models/scaler.pkl')
    print("Saved scaler to src/models/random_forest/saved_models/scaler.pkl")
    
    # Train Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42
    )
    
    rf_model.fit(X_train_scaled, y_train.values.ravel())
    
    # Save the model
    joblib.dump(rf_model, 'src/models/random_forest/saved_models/model.pkl')
    print("Saved model to src/models/random_forest/saved_models/model.pkl")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    train_random_forest() 