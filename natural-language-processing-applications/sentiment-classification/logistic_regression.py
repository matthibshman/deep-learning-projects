from feature_extractor import *
from utils import SentimentExample
from model_utils import *
from sentiment_classifier import SentimentClassifier

import numpy as np
from numpy import ndarray
from random import shuffle

from typing import List


class LogisticRegressionClassifier(SentimentClassifier):
    def __init__(self, weights: ndarray, featurizer: FeatureExtractor, threshold: float = 0.5):
        self.weights = weights
        self.featurizer = featurizer
        self.threshold = threshold

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words to classify
        :return: 0 for negative, 1 for positive
        """
        return (
            self.POSITIVE
            if logistic_probability(self.weights, self.featurizer.extract_features(sentence))
            > self.threshold
            else self.NEGATIVE
        )


def train_logistic_regression(
    train_exs: List[SentimentExample],
    feat_extractor: FeatureExtractor,
    epochs: int = 100,
    alpha=0.005,
) -> LogisticRegressionClassifier:
    """
    :return: trained LogisticRegressionClassifier model
    """
    weights = np.zeros(len(feat_extractor.get_indexer()))

    for _ in range(1, epochs):
        shuffle(train_exs)
        for example in train_exs:
            feat_vector = feat_extractor.extract_features(example.words)
            label = transform_label(example.label)
            update_weights(
                weights,
                (1.0 + label) / 2.0 - logistic_probability(weights, feat_vector),
                alpha,
                feat_vector,
            )

    return LogisticRegressionClassifier(weights, feat_extractor)
