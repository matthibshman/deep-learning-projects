from utils import Indexer
from model_utils import sentence_to_indices

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import random


class LanguageModel(object):
    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Implement in subclass")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context:
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1) ...
        :return: probability
        """
        raise Exception("Implement in subclass")


class RNNLanguageModel(LanguageModel):
    def __init__(self, vocab_index: Indexer):
        self.vocab_index = vocab_index
        self.rnn = RNNLM(len(vocab_index))
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def get_next_char_log_probs(self, context):
        with torch.no_grad():
            features = torch.tensor(sentence_to_indices(context, self.vocab_index))
            output = self.log_softmax(self.rnn.eval()(features)).numpy()
            if output.ndim == 1:
                return output
            else:
                return output[-1, :]

    def get_log_prob_sequence(self, next_chars, context):
        with torch.no_grad():
            sentence = context + next_chars
            features = torch.tensor(sentence_to_indices(sentence, self.vocab_index))
            output = self.log_softmax(self.rnn.eval()(features)).numpy()
            log_prob = 0.0
            for i in range(len(context) - 1, len(sentence) - 1):
                log_prob += output[i, self.vocab_index.index_of(sentence[i + 1])]

            return log_prob

    def train(
        self,
        train_data: str,
        num_epochs: int = 15,
        lr: float = 0.005,
        chunk_size=50,
    ):
        optimizer = optim.Adam(self.rnn.parameters(), lr=lr)
        loss = torch.nn.CrossEntropyLoss()

        indices = list(range(chunk_size, len(train_data) - 1, chunk_size))
        self.rnn.train()
        for _ in range(num_epochs):
            # total_loss = 0.0
            random.shuffle(indices)

            for idx in indices:
                features, target = self.extract_features(train_data, idx, chunk_size)

                self.rnn.zero_grad()
                output = self.rnn(features)
                curr_loss = loss(output, target)
                # total_loss += torch.sum(curr_loss)

                curr_loss.backward()
                optimizer.step()
            # print("Total loss on epoch %i: %f" % (epoch, total_loss))

    def extract_features(self, train_data: str, idx: int, chunk_size: int):
        context = train_data[idx - chunk_size : idx]
        context_shifted = train_data[idx - chunk_size + 1 : idx + 1]
        return torch.tensor(sentence_to_indices(context, self.vocab_index)), torch.tensor(
            sentence_to_indices(context_shifted, self.vocab_index)
        )


class RNNLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_size=10, hidden_size=20):
        super(RNNLM, self).__init__()
        self.hidden_size = hidden_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=2, dropout=0.2, batch_first=True)
        layers = []
        layers.append(torch.nn.Linear(hidden_size, vocab_size))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        init_state = (
            torch.zeros(2, 1, self.hidden_size),
            torch.zeros(2, 1, self.hidden_size),
        )
        embedded = self.word_embedding(x)
        output, (_, _) = self.rnn(embedded.unsqueeze(0), init_state)
        return self.network(output.squeeze())


def train_lm(train_text, vocab_index):
    """
    :param train_text: training text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    rnn_lm = RNNLanguageModel(vocab_index)
    rnn_lm.train(train_text)
    return rnn_lm
