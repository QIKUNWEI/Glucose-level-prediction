"""Train the glucose forecaster (kept for backwards compatibility).

Equivalent to ``python -m glucose_prediction.train``; all arguments are forwarded, e.g.::

    python glucose_level_prediction_model_training.py \
        --train data/559-ws-training.xml --test data/559-ws-testing.xml --out artifacts/559 --plot
"""
from glucose_prediction.train import main

if __name__ == "__main__":
    main()
