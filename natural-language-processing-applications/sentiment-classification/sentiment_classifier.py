from typing import List


class SentimentClassifier(object):
    POSITIVE = 1
    NEGATIVE = 0

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words to classify
        :return: 0 for negative, 1 for positive
        """
        raise Exception("Use subclass")
