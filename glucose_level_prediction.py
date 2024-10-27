import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import xml.etree.ElementTree as ET


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


# Preprocess the data
def preprocess_data_for_plotting(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()
    return df


# Preprocess the data for prediction
def preprocess_data_for_prediction(df, scaler):
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()

    df[['glucose_level', 'meal', 'exercise', 'insulin']] = scaler.transform(
        df[['glucose_level', 'meal', 'exercise', 'insulin']])

    return df


# Create sequences for LSTM
def create_sequences(data, sequence_length):
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :])
    return np.array(X)


# Plot recorded and predicted glucose levels with enhanced styling for high/low glucose levels
def plot_recorded_and_predicted(df_recorded, predicted_glucose, prediction_times):
    recorded_7d = df_recorded.tail(7 * 24 * 12)  # last 7 days of recorded data

    plt.figure(figsize=(12, 6))

    # Plot recorded glucose levels (last 7 days)
    plt.plot(recorded_7d.index, recorded_7d['glucose_level'], label='Recorded Glucose Levels (Last 7 Days)', color='#BFB7A8')

    # Plot predicted glucose levels with thicker lines for high/low regions
    low_threshold = 70
    high_threshold = 180

    # Initialize flags to ensure labels are shown only once
    low_glucose_plotted = False
    high_glucose_plotted = False
    normal_glucose_plotted = False

    # Iterate through predicted glucose levels and color them based on concentration
    for i in range(len(prediction_times) - 1):
        glucose_level = predicted_glucose[i]

        if glucose_level < low_threshold:
            # Plot low glucose (thicker line) and ensure label is added only once
            plt.plot(prediction_times[i:i + 2], predicted_glucose[i:i + 2], color='purple', linewidth=2.5,
                     label='Hypoglycemia risk (< 70 mg/dL)' if not low_glucose_plotted else "")
            low_glucose_plotted = True
        elif glucose_level > high_threshold:
            # Plot high glucose (thicker line) and ensure label is added only once
            plt.plot(prediction_times[i:i + 2], predicted_glucose[i:i + 2], color='red', linewidth=2.5,
                     label='Hyperglycemia risk (> 180 mg/dL)' if not high_glucose_plotted else "")
            high_glucose_plotted = True
        else:
            # Plot normal glucose (regular line) and ensure label is added only once
            plt.plot(prediction_times[i:i + 2], predicted_glucose[i:i + 2], color='blue', linewidth=1.5)
            normal_glucose_plotted = True

    plt.xlabel('Time')
    plt.ylim(0, 350)
    plt.ylabel('Glucose Level (mg/dL)')
    plt.title('Recorded Glucose Levels (Last 7 Days) and Predicted Glucose Levels (Next 48 Hours)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_evaluation_metrics(y_true, y_pred, times):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

    plt.figure(figsize=(12, 6))

    # Plot true and predicted glucose levels
    plt.plot(times, y_true, label='True Glucose Levels', color='blue')
    plt.plot(times, y_pred, label='Predicted Glucose Levels', linestyle='--', color='orange')

    plt.title('True vs Predicted Glucose Levels')
    plt.ylim(0, 350)
    plt.xlabel('Time')
    plt.ylabel('Glucose Level (mg/dL)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Predict glucose levels for the next 48 hours
def predict_next_48_hours(file_path, model=None, sequence_length=12):
    df_new_patient = load_xml_data(file_path)  # Load the XML data from the test file

    df_plot = preprocess_data_for_plotting(df_new_patient.copy())

    if model is None:
        raise ValueError("A trained model must be provided to predict glucose levels.")

    scaler = joblib.load('scaler.pkl')  # Load the scaler saved during training
    df_new_patient = preprocess_data_for_prediction(df_new_patient, scaler)

    data = df_new_patient.values
    X_new = create_sequences(data, sequence_length)

    predicted_glucose = model.predict(X_new)

    predicted_glucose_rescaled = scaler.inverse_transform(
        np.concatenate([predicted_glucose, np.zeros((predicted_glucose.shape[0], 3))], axis=1))[:, 0]

    last_timestamp = df_plot.index[-1]
    prediction_times = pd.date_range(last_timestamp, periods=len(predicted_glucose), freq='5min')

    if len(df_plot) > len(predicted_glucose_rescaled):
        y_true = df_plot['glucose_level'].values[-len(predicted_glucose_rescaled):]
        plot_evaluation_metrics(y_true, predicted_glucose_rescaled, prediction_times)

    return df_plot, predicted_glucose_rescaled[:576], prediction_times[:576]


if __name__ == "__main__":
    new_patient_file = 'path to the test file'
    trained_model = load_model('path to trained h5 model file')

    df_recorded, predicted_glucose, prediction_times = predict_next_48_hours(new_patient_file, model=trained_model)

    # Plot both the recorded glucose levels (last 7 days) and the predicted glucose levels (next 48 hours)
    plot_recorded_and_predicted(df_recorded, predicted_glucose, prediction_times)
