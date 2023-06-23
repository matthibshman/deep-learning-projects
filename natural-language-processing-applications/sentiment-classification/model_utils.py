from feature_extractor import *

import numpy as np
from numpy import ndarray

from collections import Counter


def dot_product(w: ndarray, f: Counter) -> float:
    sum = 0.0
    for (index, value) in f.items():
        sum += w[index] * value
    return sum


def update_weights(weights: ndarray, label: float, alpha: float, feat_vector: Counter):
    for (index, _) in feat_vector.items():
        weights[index] += label * alpha


def logistic_probability(w: ndarray, f: Counter) -> float:
    e = np.exp(dot_product(w, f))
    return e / (1.0 + e)


def transform_label(label: int) -> float:
    return 1.0 if label == 1 else -1.0
