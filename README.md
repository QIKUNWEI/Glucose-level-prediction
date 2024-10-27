**Glucose Level Prediction Model**

This repository contains code for predicting blood glucose levels using a time-series machine learning model, developed as part of a data-driven life sciences project. The model is designed to assist in Type 1 diabetes management by forecasting glucose fluctuations, providing early warnings for hyperglycemia and hypoglycemia, and facilitating proactive interventions.

**Project Overview**

This project leverages the OhioT1DM dataset and applies a Bidirectional LSTM model to capture temporal patterns in glucose data. The model uses timestamped glucose readings to predict future levels, helping patients or automated systems to make timely adjustments to insulin dosing.

**Features**
1. Data Preprocessing: Normalizes and sequences glucose data for model input.
2. Bidirectional LSTM Model: Predicts glucose levels based on historical timestamped data.
3. 	Early Warning System: Identifies high and low glucose risks, supporting timely interventions.

**Dataset**

The model uses the publicly available OhioT1DM dataset. For access, visit [Kaggle](https://www.kaggle.com/datasets/ryanmouton/ohiot1dm/data).

**Usage**

1. Clone the repository.

2. Run glucose_level_prediction.py to train the model.
  
3. Use glucose_level_prediction_model_training.py for prediction and visualization of results.

**License**

This project is licensed under the MIT License. See the LICENSE file for details.
