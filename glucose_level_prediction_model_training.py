import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt
import joblib

# Load XML data and parse glucose levels, meal, exercise, and insulin data
def load_xml_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    events = []

    for event in root.findall('.//glucose_level/event'):
        timestamp = event.get('ts')
        glucose_value = float(event.get('value'))
        meal = 0
        exercise = 0
        insulin = 0

        meal_event = event.find('.//meal')
        if meal_event is not None:
            meal = float(meal_event.get('carbs'))

        exercise_event = event.find('.//exercise')
        if exercise_event is not None:
            exercise = float(exercise_event.get('intensity'))

        insulin_event = event.find('.//insulin')
        if insulin_event is not None:
            insulin = float(insulin_event.get('dose'))

        events.append((timestamp, glucose_value, meal, exercise, insulin))

    df = pd.DataFrame(events, columns=['timestamp', 'glucose_level', 'meal', 'exercise', 'insulin'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M:%S')

    return df

# Preprocess the data (scaling and preparing for LSTM)
def preprocess_data(df):
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()

    scaler = MinMaxScaler()
    df[['glucose_level', 'meal', 'exercise', 'insulin']] = scaler.fit_transform(df[['glucose_level', 'meal', 'exercise', 'insulin']])

    joblib.dump(scaler, 'scaler.pkl')

    return df, scaler

# Create sequences for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

# Build the LSTM model
def build_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='mean_squared_error',
                  metrics=[RootMeanSquaredError(name='rmse'), MeanAbsoluteError(name='mae')])

    return model

# Function to plot training metrics
def plot_training_metrics(history):
    rmse = history.history['rmse']
    val_rmse = history.history['val_rmse']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(rmse) + 1)

    # Plot RMSE
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, rmse, label='Training RMSE')
    plt.plot(epochs, val_rmse, label='Validation RMSE')
    plt.title('Training and Validation RMSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot MAE
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, mae, label='Training MAE')
    plt.plot(epochs, val_mae, label='Validation MAE')
    plt.title('Training and Validation MAE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to train the model
def train_model(file_path, sequence_length=12, test_size=0.2):
    df = load_xml_data(file_path)

    df, scaler = preprocess_data(df)

    data = df.values
    X, y = create_sequences(data, sequence_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    model = build_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    mse, rmse, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {mse}, Test RMSE: {rmse}, Test MAE: {mae}")

    save_model(model, 'glucose_model_trained.h5')

    return model, scaler, X_test, y_test, history

if __name__ == "__main__":
    train_file_path = 'path to training set'
    model, scaler, X_test, y_test, history = train_model(train_file_path)

    # Plot training metrics
    plot_training_metrics(history)
