import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_split_data_lstm(filepath):
    """Reusable function to load, normalise, and split data for LSTM RNN model."""
    data = pd.read_csv(filepath)
    # Convert angles to one of a continuous scale (radian)
    data['INITIAL SHUTTLECOCK FIRING ANGLE(DEGREE)'] = np.radians(data['INITIAL SHUTTLECOCK FIRING ANGLE(DEGREE)'])
    data['INITIAL SHUTTLECOCK SLANT ANGLE TO SIDELINE(DEGREE)'] = np.radians(data['INITIAL SHUTTLECOCK SLANT ANGLE TO SIDELINE(DEGREE)'])
    # Rename columns
    data = data.rename(columns={
        'INITIAL SHUTTLECOCK FIRING ANGLE(DEGREE)': 'INITIAL SHUTTLECOCK FIRING ANGLE(RADIAN)',
        'INITIAL SHUTTLECOCK SLANT ANGLE TO SIDELINE(DEGREE)': 'INITIAL SHUTTLECOCK SLANT ANGLE TO SIDELINE(RADIAN)'})
    
    # Get unique numbers of observation groups
    observation_grps = data['OBSERVATION GROUP NUMBER'].unique()
    # Split observations groups into train-test sets, 80-20
    train_grps, test_grps = train_test_split(observation_grps, test_size=0.2, random_state=42)
    train_data = data[data['OBSERVATION GROUP NUMBER'].isin(train_grps)]
    test_data = data[data['OBSERVATION GROUP NUMBER'].isin(test_grps)]
    # Select features (X) and target variables (Y)
    x_columns = ['HUMAN PLAYER POSITION(X) metres',
        'HUMAN PLAYER POSITION(Y) metres',
        'HUMAN PLAYER POSITION(Z) metres',
        'INITIAL VELOCITY OF SHUTTLECOCK(m/s)',
        'INITIAL SHUTTLECOCK FIRING ANGLE(RADIAN)', 
        'INITIAL SHUTTLECOCK SLANT ANGLE TO SIDELINE(RADIAN)',
        'OBSERVATION NUMBER']
    y_columns = ['SHUTTLECOCK POSITION IN AIR(X) metres', 
        'SHUTTLECOCK POSITION IN AIR(Y) metres', 
        'SHUTTLECOCK POSITION IN AIR(Z) metres']
    x_train, x_test = train_data[x_columns], test_data[x_columns]
    y_train, y_test = train_data[y_columns], test_data[y_columns]
    # Initialize MinMaxScaler
    scaler_X, scaler_Y = MinMaxScaler(), MinMaxScaler()
    
    # Normalize X and Y
    X_train_scaled, X_test_scaled = scaler_X.fit_transform(x_train), scaler_X.transform(x_test)
    Y_train_scaled, Y_test_scaled = scaler_Y.fit_transform(y_train), scaler_Y.transform(y_test)

    # Reshaping training data to fit LSTM input requirement (samples, time_steps, features)
    X_train_scaled_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Print Dataset Summary
    print(f"Data Loaded & Processed\n"
          f"Total observation groups: {len(observation_grps)}\n"
          f"Training groups: {len(train_grps)}, Testing groups: {len(test_grps)}\n"
          f"Training samples: {X_train_scaled_reshaped.shape[0]}, Testing samples: {X_test_scaled_reshaped.shape[0]}\n"
          f"X Features: {X_train_scaled_reshaped.shape[2]}, Y Features: {Y_train_scaled.shape[1]}\n")

    return X_train_scaled_reshaped, X_test_scaled_reshaped, Y_train_scaled, Y_test_scaled, scaler_X, scaler_Y

def build_lstm_model(lstm_units_1, lstm_units_2, input_shape):
    """Reusable function to build LSTM RNN model with varying units for layer 1 and 2."""
    model = Sequential()
    # Builds LSTM RNN with two main layers, with varying units for layer 1 and 2
    model.add(LSTM(units=lstm_units_1, activation='tanh', return_sequences=True, input_shape=input_shape))      # First LSTM Layer
    model.add(LSTM(units=lstm_units_2, activation='tanh'))                                                      # Second LSTM Layer
    model.add(Dense(3, activation='linear'))                                                                    # Activation layer, to predict X, Y, Z positions
    # Compile and return model
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return model

def train_lstm_rnn_model(filepath, lstm_units_1, lstm_units_2, batch_size, outpath):
    """Reusable function to train an LSTM RNN model."""
    x_train_reshaped, x_test_reshaped, y_train, y_test, scaler_X, scaler_Y = load_and_split_data_lstm(filepath)
    lstm_rnn_model = build_lstm_model(lstm_units_1, lstm_units_2, (0, x_train_reshaped.shape[2]))

    # Print model summary before training
    print("✅ Model Architecture:")
    lstm_rnn_model.summary()

    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)    
    lstm_rnn_model.fit(x_train_reshaped, 
        y_train, 
        epochs=50, 
        batch_size=batch_size, 
        validation_split=0.1,
        callbacks=[early_stopping], 
        verbose=1)
    # Save trained model in `.keras` format
    os.makedirs('models/trained_models', exist_ok=True)
    lstm_rnn_model.save(outpath)
    # Save scalers for future use
    joblib.dump(scaler_X, outpath.replace('.keras', '_scaler_X.pkl'))
    joblib.dump(scaler_Y, outpath.replace('.keras', '_scaler_Y.pkl'))
    print(f"✅ Model saved as {outpath}")
    print(f"✅ Scalers saved as {outpath.replace('.keras', '_scaler_X.pkl')} & {outpath.replace('.keras', '_scaler_Y.pkl')}")