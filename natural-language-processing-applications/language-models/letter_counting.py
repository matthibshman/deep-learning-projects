import math
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim

from utils import *


# Outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size=27,
        d_model=90,
        d_internal=75,
        num_classes=3,
        num_layers=2,
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.transformer_layers = []
        for _ in range(num_layers):
            self.transformer_layers.append(TransformerLayer(d_model, d_internal))

        class_layers = []
        # class_layers.append(torch.nn.Linear(d_model, d_model))
        # class_layers.append(torch.nn.ReLU())
        class_layers.append(torch.nn.Linear(d_model, num_classes))
        # class_layers[-1].weight.zero_
        self.network = torch.nn.Sequential(*class_layers)
        # self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, indices):
        """
        :return: tuple of  softmax log probabilities and list of attention maps used in neural net layers
        """
        attention_maps = []

        inputs = self.positional_encoding(self.word_embedding(indices))
        for transformer_layer in self.transformer_layers:
            output, attention_map = transformer_layer(inputs)
            inputs = output
            attention_maps.append(attention_map)

        return self.network(inputs), attention_maps


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer
        :param d_internal: The "internal" dimension used in the self-attention computation
        """
        super().__init__()
        self.d_model = d_model
        self.soft_max = torch.nn.Softmax(0)
        self.w_Q = torch.nn.Linear(d_model, d_internal)
        self.w_K = torch.nn.Linear(d_model, d_internal)
        self.w_V = torch.nn.Linear(d_model, d_internal)
        self.w_0 = torch.nn.Linear(d_internal, d_model)

        ff_layers = []
        ff_layers.append(torch.nn.Linear(d_model, d_model * 2))
        ff_layers.append(torch.nn.ReLU())
        ff_layers.append(torch.nn.Linear(d_model * 2, d_model))
        self.ff_net = torch.nn.Sequential(*ff_layers)

    def forward(self, input_vecs):
        Q = self.w_Q(input_vecs)
        K = self.w_K(input_vecs)
        V = self.w_V(input_vecs)
        self_attention_map = self.soft_max(
            torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.d_model)
        )
        self_attention = self.w_0(torch.matmul(self_attention_map, V))
        residuals = self_attention + input_vecs

        feed_forward = self.ff_net(residuals)

        return feed_forward + residuals, self_attention_map


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20):
        """
        :param d_model: dimensionality of the embedding layer
        :param num_positions: the maximum sequence length this module will see
        """
        super().__init__()
        self.emb = nn.Embedding(num_positions, d_model)

    def forward(self, x):
        """
        :return: a tensor of the same size with positional embeddings added in
        """
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        return x + self.emb(indices_to_embed)


def train_classifier(train: list[LetterCountingExample]):
    model = Transformer()
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
    # )
    loss_fcn = nn.CrossEntropyLoss()

    num_epochs = 10
    for _ in range(0, num_epochs):
        # loss_this_epoch = 0.0
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)

        for ex_idx in ex_idxs:
            sample = train[ex_idx]
            features, target = sample.input_tensor, sample.output_tensor

            model.zero_grad()
            output, _ = model(features)
            loss = loss_fcn(output, target)
            # loss_this_epoch += loss.item()

            loss.backward()
            optimizer.step()
        # print("Total loss on epoch %i: %f" % (t, loss_this_epoch))

    model.eval()
    return model
