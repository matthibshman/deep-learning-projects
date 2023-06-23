from utils import Indexer


def sentence_to_indices(sentence: str, vocab_index: Indexer):
    return [vocab_index.index_of(char) for char in sentence]
