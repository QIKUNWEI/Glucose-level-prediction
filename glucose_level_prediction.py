"""Forecast glucose with a trained model (kept for backwards compatibility).

Equivalent to ``python -m glucose_prediction.predict``; all arguments are forwarded, e.g.::

    python glucose_level_prediction.py --model artifacts/559 \
        --input data/559-ws-testing.xml --history data/559-ws-training.xml --plot forecast.png
"""
from glucose_prediction.predict import main

if __name__ == "__main__":
    main()
