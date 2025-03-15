import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

def create_q_model(input_dim=6, output_dim=6):
    """
    Creates and returns a simple Q-network model.
    The network has an explicit Input layer (to avoid warnings),
    two hidden Dense layers with 64 units each, and a linear output layer.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
    return model

if __name__ == '__main__':
    # For testing purposes, print a summary of the model.
    model = create_q_model()
    model.summary()
