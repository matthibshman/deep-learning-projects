from utils import Indexer
from model_utils import sentence_to_indices

from typing import List
import torch
import torch.nn as nn
from torch import optim
import random


class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Implement in subclass")


class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, vocab_index: Indexer):
        self.vocab_index = vocab_index
        self.rnn = RNN(len(vocab_index))
        self.softmax = nn.Softmax()

    def predict(self, context):
        with torch.no_grad():
            output = self.rnn.eval()(torch.tensor(sentence_to_indices(context, self.vocab_index)))
            return torch.argmax(self.softmax(output)).item()

    def train(
        self,
        train_vowels: List[str],
        train_cons: List[str],
        vowel_label=1,
        cons_label=0,
        num_epochs: int = 10,
        lr: float = 0.004,
    ):
        optimizer = optim.Adam(self.rnn.parameters(), lr=lr)
        loss = torch.nn.CrossEntropyLoss()

        indices = list(range(len(train_vowels) + len(train_cons)))
        self.rnn.train()
        for _ in range(num_epochs):
            # total_loss = 0.0
            random.shuffle(indices)
            for idx in indices:
                features, label = self.extract_features(
                    idx, train_vowels, train_cons, vowel_label, cons_label
                )

                self.rnn.zero_grad()
                output = self.rnn(features)
                curr_loss = loss(output.unsqueeze(0), label)
                # total_loss += torch.sum(curr_loss)

                curr_loss.backward()
                optimizer.step()
            # print("Total loss on epoch %i: %f" % (epoch, total_loss))

    def extract_features(
        self,
        idx: int,
        train_vowels: list[str],
        train_cons: list[str],
        vowel_label: int,
        cons_label: int,
    ):
        label = vowel_label
        if idx < len(train_vowels):
            sentence = train_vowels[idx]
        else:
            sentence = train_cons[idx - len(train_vowels)]
            label = cons_label

        return torch.tensor(sentence_to_indices(sentence, self.vocab_index)), torch.tensor([label])


class RNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_size=10, hidden_size=10, output_classes=2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=1, dropout=0, batch_first=True)
        layers = []
        layers.append(torch.nn.Linear(hidden_size, output_classes))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        init_state = (
            torch.zeros(1, 1, self.hidden_size),
            torch.zeros(1, 1, self.hidden_size),
        )
        embedded = self.word_embedding(x)
        _, (hidden_state, _) = self.rnn(embedded.unsqueeze(0), init_state)
        return self.network(hidden_state.squeeze())


def train_rnn_classifier(train_cons_exs, train_vowel_exs, vocab_index):
    """
    :param train_cons_exs: strings followed by consonants
    :param train_vowel_exs: strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    rnn_classifier = RNNClassifier(vocab_index)
    rnn_classifier.train(train_vowel_exs, train_cons_exs)
    return rnn_classifier
