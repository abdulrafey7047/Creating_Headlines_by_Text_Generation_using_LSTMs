import numpy as np

from typing import Dict

def generate_embedding_matrix_from_file(file_path: str, vocab_size: int, embedding_dim: int,
                                        token_word_index: Dict) -> np.ndarray:
    """
    Function to generate a numpy matrix of GloVe embeddings
    given the embedding file path.
    NOTE: This function only parsees the format of GloVe
          embeddings, it will not work on any other format.

    @args
    filepath:           str, path to embedding file
    vocab_size:         int, number of words in vocabulary
    embedding_dim:      int, dimesion for embedding of each word
    token_word_index:   dict, map of word to their int tokens

    @returns
    embedding_matrix:   np.ndarray, the embedding matrix of shape
                        (vocab_size, embedding_dim)
    hits:               int, the number of words from vocab whose
                        embeddings were found in the embedding file
    misses:             int, the number of words from vocab whose
                        embeddings were not found in the embedding
                        file
    """

    embeddings_index = dict()
    with open(file_path) as f:
        for line in f:
            word, embedding = line.split(maxsplit=1)
            embedding = np.fromstring(embedding, "f", sep=" ")
            embeddings_index[word] = embedding


    hits, misses = 0, 0
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, index in token_word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            hits += 1
        else:
            misses += 1

    return embedding_matrix, hits, misses
