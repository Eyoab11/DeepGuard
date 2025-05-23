import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_test_random_forest():
    # Create dummy training data
    n_samples = 1000
    n_features = 28  # Number of features in our config
    
    # Generate random feature data
    X = np.random.randn(n_samples, n_features)
    
    # Generate random binary labels
    y = np.random.randint(0, 2, n_samples)
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    
    rf_model.fit(X_scaled, y)
    
    # Save model and scaler
    os.makedirs('src/models/random_forest/saved_models', exist_ok=True)
    joblib.dump(rf_model, 'src/models/random_forest/saved_models/model.pkl')
    joblib.dump(scaler, 'src/models/random_forest/saved_models/scaler.pkl')
    
    print("Created and saved test Random Forest model and scaler")

if __name__ == "__main__":
    create_test_random_forest() 