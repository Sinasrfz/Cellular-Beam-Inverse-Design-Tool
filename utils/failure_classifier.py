# utils/failure_classifier.py

import numpy as np
import pandas as pd


def predict_failure_mode(clf, encoder, X_scaled):
    pred = clf.predict(X_scaled)
    label = encoder.inverse_transform(pred)[0]
    return label
