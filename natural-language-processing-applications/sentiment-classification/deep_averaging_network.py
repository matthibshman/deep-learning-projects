from feature_extractor import *
from utils import SentimentExample
from model_utils import *
from sentiment_classifier import SentimentClassifier

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

MISSING_INDEX = -1
MISSING_WORD = "UNK"


class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, word_embeddings: WordEmbeddings):
        self.word_embeddings = word_embeddings
        self.dan = DeepAveragingNetwork(word_embeddings.get_initialized_embedding_layer())

    def predict(self, sentence: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param sentence: words to classify
        :return: 0 for negative, 1 for positive
        """
        with torch.no_grad():
            output = self.dan.eval()(torch.IntTensor([get_indices(sentence, self.word_embeddings)]))

            return int(torch.argmax(output).item())

    def train(
        self,
        train_exs: List[SentimentExample],
        num_epochs: int = 25,
        lr: float = 0.02,
        batch_size: int = 64,
    ):
        optimizer = optim.Adam(self.dan.parameters(), lr=lr)
        loss = torch.nn.CrossEntropyLoss()

        sentiment_dataloader = DataLoader(
            SentimentDataDataset(train_exs, self.word_embeddings),
            num_workers=1,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        for epoch in range(num_epochs):
            # total_loss = 0.0
            for features, labels in sentiment_dataloader:
                self.dan.zero_grad()
                output = self.dan(features)
                curr_loss = loss(output, labels)
                # total_loss += torch.sum(curr_loss)

                curr_loss.backward()
                optimizer.step()
            # print("Total loss on epoch %i: %f" % (epoch, total_loss))


def get_indices(sentence: List[str], word_embeddings: WordEmbeddings):
    return [
        index if index != MISSING_INDEX else word_embeddings.word_indexer.index_of(MISSING_WORD)
        for index in [word_embeddings.word_indexer.index_of(word) for word in sentence]
    ]


class SentimentDataDataset(Dataset):
    def __init__(self, samples: List[SentimentExample], word_embeddings: WordEmbeddings):
        self.FEATURE_LENGTH = 128
        self.PAD_VALUE = 0

        self.features = np.empty([len(samples), self.FEATURE_LENGTH], dtype=int)
        labels = [0] * len(samples)

        for i, sentiment_example in enumerate(samples):
            sentiment_example_indices = get_indices(sentiment_example.words, word_embeddings)
            sample = self.normalizeSampleSize(sentiment_example_indices)
            self.features[i] = sample
            labels[i] = sentiment_example.label

        self.labels = np.array(labels)
        self.size = len(self.features)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError

        return (self.features[idx], self.labels[idx])

    def normalizeSampleSize(self, current_sample: List[int]):
        sample = np.array(current_sample)
        if sample.size < self.FEATURE_LENGTH:
            sample = np.pad(
                sample,
                (0, self.FEATURE_LENGTH - sample.size),
                "constant",
                constant_values=(self.PAD_VALUE),
            )
        elif sample.size > self.FEATURE_LENGTH:
            sample = sample[: self.FEATURE_LENGTH]
        return sample


class DeepAveragingNetwork(nn.Module):
    def __init__(
        self,
        embedding_layer: torch.nn.Embedding,
        dropout_rate: float = 0.1,
        output_classes: int = 3,
    ):
        super(DeepAveragingNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.embedding_layer.padding_idx = 0
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        hidden_layer_size = embedding_layer.embedding_dim
        layers = []
        layers.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_layer_size, output_classes))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        embedded = torch.mean(self.embedding_layer(x), dim=1)
        return self.network(self.dropout(embedded))


def train_deep_averaging_network(
    train_exs: List[SentimentExample],
    word_embeddings: WordEmbeddings,
) -> NeuralSentimentClassifier:
    """
    :return: A trained NeuralSentimentClassifier model
    """
    classifier = NeuralSentimentClassifier(word_embeddings)
    classifier.train(train_exs)
    return classifier
