from utils import *
from collections import Counter
from typing import List

import torch


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Use subclass")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        raise Exception("Use subclass")

    def is_in_vocab(self, index):
        return index != -1


class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer, train_exs: List[SentimentExample]):
        self.indexer = indexer
        # construct vocabulary from training examples
        if train_exs is not None:
            for example in train_exs:
                for word in example.words:
                    self.indexer.add_and_get_index(word.lower())

    def get_indexer(self) -> Indexer:
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        indices = []
        for token in sentence:
            index = self.indexer.add_and_get_index(token.lower(), add_to_indexer)
            # only consider features in vocabulary
            if self.is_in_vocab(index):
                indices.append(index)

        return Counter(indices)


class BigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer, train_exs: List[SentimentExample]):
        self.indexer = indexer
        # construct vocabulary from training examples
        if train_exs is not None:
            for example in train_exs:
                for i in range(len(example.words) - 1):
                    self.indexer.add_and_get_index(
                        example.words[i].lower() + "|" + example.words[i + 1].lower()
                    )

    def get_indexer(self) -> Indexer:
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        indices = []
        for i in range(len(sentence) - 1):
            index = self.indexer.add_and_get_index(
                sentence[i].lower() + "|" + sentence[i + 1].lower(), add_to_indexer
            )
            # only consider features in vocabulary
            if self.is_in_vocab(index):
                indices.append(index)

        return Counter(indices)


class BetterFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer, stop_words: set[str], train_exs: List[SentimentExample]):
        self.indexer = indexer
        self.n_gram_size = 2

        if train_exs is not None:
            # top_n_words = self.build_top_N_words(100000, train_exs, stop_words)

            for example in train_exs:
                lowered_example = [word.lower() for word in example.words]
                filtered_example = [
                    word
                    for word in lowered_example
                    # if word not in stop_words
                ]
                for i in range(1, self.n_gram_size + 1):
                    self.build_n_gram(filtered_example, i)

    def get_indexer(self) -> Indexer:
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        counter = Counter()
        for i in range(1, self.n_gram_size):
            counter.update(self.build_n_gram(sentence, i, add_to_indexer, True))

        return counter

    def build_n_gram(
        self,
        sentence: list[str],
        n: int,
        add_to_indexer: bool = True,
        with_indices: bool = False,
    ):
        indices = []

        for i in range(len(sentence) - (n - 1)):
            n_gram = sentence[i].lower()
            for j in range(1, n):
                n_gram += "|" + sentence[i + j].lower()

            index = self.indexer.add_and_get_index(
                n_gram,
                add_to_indexer,
            )
            if self.is_in_vocab(index) and with_indices:
                indices.append(index)

        return indices

    # build set of top-N words, filtering out stop words
    def build_top_N_words(
        self, N: int, train_exs: List[SentimentExample], stop_words: set[str]
    ) -> set[str]:
        beam = Beam(N)
        counter = Counter()

        for example in train_exs:
            counter.update([word.lower() for word in example.words if word not in stop_words])
        for word, count in counter.items():
            beam.add(word, count)
        return set(beam.get_elts())
    
class WordEmbeddings:
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_initialized_embedding_layer(self):
        return torch.nn.Embedding.from_pretrained(torch.FloatTensor(self.vectors))

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]
