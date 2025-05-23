import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Create a simple model for testing
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, 10)),  # Assuming 10 features after preprocessing
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the model
model.save('src/models/lstm/saved_models/latest.h5')

# Create and save a test scaler
scaler = StandardScaler()
# Fit with some dummy data
dummy_data = np.random.randn(100, 10)
scaler.fit(dummy_data)
joblib.dump(scaler, 'src/models/lstm/saved_models/scaler.pkl') 