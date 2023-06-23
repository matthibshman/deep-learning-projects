from feature_extractor import *
from utils import SentimentExample
from model_utils import *
from sentiment_classifier import SentimentClassifier


import numpy as np
from numpy import ndarray
from random import shuffle

from typing import List

class PerceptronClassifier(SentimentClassifier):
    def __init__(self, weights: ndarray, featurizer: FeatureExtractor):
        self.weights = weights
        self.featurizer = featurizer

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words to classify
        :return: 0 for negative, 1 for positive
        """
        return (
            self.POSITIVE
            if dot_product(self.weights, self.featurizer.extract_features(sentence)) > 0
            else self.NEGATIVE
        )
    
def train_perceptron(
    train_exs: List[SentimentExample],
    feat_extractor: FeatureExtractor,
    epochs: int = 150,
    alpha=0.01,
) -> PerceptronClassifier:
    """
    :return: trained PerceptronClassifier model
    """
    weights = np.zeros(len(feat_extractor.get_indexer()))

    for _ in range(epochs):
        shuffle(train_exs)
        for example in train_exs:
            feat_vector = feat_extractor.extract_features(example.words)
            y_pred = 1 if dot_product(weights, feat_vector) > 0 else 0
            if y_pred != example.label:
                label = transform_label(example.label)
                update_weights(weights, label, alpha, feat_vector)

    return PerceptronClassifier(weights, feat_extractor)