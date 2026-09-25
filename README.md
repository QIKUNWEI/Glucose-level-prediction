**Glucose Level Prediction Model**

This repository forecasts blood glucose 5–60 minutes ahead for people with type 1 diabetes. It uses continuous glucose monitor (CGM) data together with insulin, meal and exercise logs from the OhioT1DM dataset. The forecasts give early warning of hypoglycaemia (< 70 mg/dL) and hyperglycaemia (> 180 mg/dL).

**Pipeline**

1. **Parsing** (`glucose_prediction/data.py`): reads every OhioT1DM stream from its own XML section. These are CGM, basal rate plus temporary basals, boluses (square/dual boluses spread over their delivery time), meals and exercise. The streams go onto a regular 5-minute grid. CGM readings snap to the nearest grid point, and missing readings stay missing.
2. **Features** (`glucose_prediction/features.py`): glucose, rate of change, an imputation flag, insulin on board and insulin activity (exponential insulin action model, 6 h duration, 75 min peak), carbs on board and carb absorption, bolus, carbs, basal rate, exercise intensity, and time of day (sin/cos). Every feature is causal. Only interior CGM gaps of 30 min or less are interpolated, and only for inputs. Targets are always real CGM readings.
3. **Model** (`glucose_prediction/model.py`): an ensemble of 5 two-layer GRU networks, plus a skip connection from the latest time step. From a 2-hour input window it predicts the whole 5–60 min trajectory at once. The networks learn the *change* from the current glucose, trained with a Huber loss that ignores missing targets. Early stopping and learning-rate decay run on a chronological validation split. The spread across ensemble members is reported as a rough uncertainty.
4. **Evaluation** (`glucose_prediction/metrics.py`, `pipeline.py`): RMSE, MAE, MARD, bias, Clarke error grid zones, and the sensitivity/precision of hypo/hyper alerts at 30 and 60 min. Every result is compared with persistence, linear extrapolation and a ridge ARX model on the same features.

**Evaluation protocol**

* The training file of each patient is split **in time**: the last 20 % is validation. Training windows whose targets would reach into the validation period are dropped (a 60-minute embargo).
* The input scaling is fitted on the training windows only.
* Hyperparameters were chosen on the validation split. The OhioT1DM *testing* file is used only for the final numbers. The end of the training file serves as history for the first test forecasts, as in the OhioT1DM challenge.
* Forecasts are made only at times with a real CGM reading and a complete input window. All models are scored on exactly the same samples.

**Results (patient 559, OhioT1DM test file, 10 days)**

Benchmark numbers will be added here once the evaluation runs finish.

**Dataset**

The model uses the OhioT1DM dataset, which you can get from [Kaggle](https://www.kaggle.com/datasets/ryanmouton/ohiot1dm/data). Place the XML files in `data/`, which is git-ignored. The dataset's data use agreement does not allow redistribution.

**Usage**

```bash
pip install -r requirements.txt

# Train on one or more patients and evaluate on the held-out test files
python -m glucose_prediction.train \
    --train data/559-ws-training.xml \
    --test data/559-ws-testing.xml \
    --out artifacts/559 --plot

# Optional: add other patients' data as extra training data only
#   --extra-train data/563-ws-training.xml data/570-ws-training.xml ...

# Forecast, raise alerts, export to CSV and plot
python -m glucose_prediction.predict \
    --model artifacts/559 \
    --input data/559-ws-testing.xml --history data/559-ws-training.xml \
    --output forecasts.csv --plot forecast.png

# Tests (synthetic data, no dataset needed)
python -m pytest tests
```

`glucose_level_prediction_model_training.py` and `glucose_level_prediction.py` are still available and forward their arguments to the two commands above. Run either module with `--help` to see every option (architecture, history length, horizon, ensemble size and so on).

**Limitations**

Forecasts beyond about an hour are not meaningful without knowing future meals and insulin, so the model predicts at most 60 minutes ahead. The earlier version's "48-hour prediction" only relabelled one-step-ahead predictions with future timestamps. This is a research prototype and not a medical device.

**License**

This project is licensed under the MIT License. See the LICENSE file for details.
