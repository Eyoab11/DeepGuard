import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create a simple LSTM model for testing
def create_test_lstm():
    # Define model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(10, 5)),  # 10 timesteps, 5 features
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Create some dummy data to fit the scaler
    dummy_data = np.random.randn(100, 5)  # 100 samples, 5 features
    scaler = StandardScaler()
    scaler.fit(dummy_data)

    # Save model and scaler
    os.makedirs('src/models/lstm/saved_models', exist_ok=True)
    model.save('src/models/lstm/saved_models/latest.h5')
    joblib.dump(scaler, 'src/models/lstm/saved_models/scaler.pkl')

    print("Created and saved test LSTM model and scaler")

if __name__ == "__main__":
    create_test_lstm() 