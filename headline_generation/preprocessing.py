import os
import re
import numpy as np
import pandas as pd

from typing import List, Tuple, Set

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataPreProcessor:

    def __init__(self, data: List):
        self.data = pd.Series(data)
        self.vocab = None
        self.tokenizer = None
        self.preprocessed_data = None

    def clean_data(self):
        """
        @returns
        self: DataPreProcessor, object with cleaned data in its 'data' attribute
        """

        ## Converting to Lowercase
        self.data = self.data.apply(str.lower)

        ## Removing Punctuations
        self.data = self.data.apply(
            lambda headline: re.sub(r'[^\w\s]', '', headline))

        return self
    
    def preprocess(self, oov_token, max_padding_len, padding_type, tokenizing_filters=[]):
        """
        Fucntion to preprocess data and bring it into a format that can be fed to the
        HeadlineGenerator class

        @args
        oov_token:          str, out of vocabulary token, passed to tensorflow's tokenizer
        max_padding_len:    int, length to which all sequences will be padded to
        padding_type:       str, where to add padding tokens, either one from 'pre' or 'post'
        tokenizing_filters: list, filters passed to tensorflow's tokenizer

        @returns
        self: DataPreProcessor, object with preprocessed data in its 'preprocessed' attribute
        """

        ## Adding <START> and <END> tokens
        self.data = self.data.apply(
            lambda headline: f'<START> {headline} <END>')

        ## Generating n_grams
        n_grams = np.array([])
        for headline in self.data:
            n_grams = np.append(n_grams, self._generate_n_grams(headline))

        ## Generating Vocabulary
        self.vocab = self.get_vocab()

        ## Tokenizing
        self.tokenizer = Tokenizer(oov_token=oov_token, filters=tokenizing_filters, lower=False)
        self.tokenizer.fit_on_texts(self.vocab)
        tokenized_n_grams = self.tokenizer.texts_to_sequences(n_grams)

        ## Padding
        self.preprocessed_data = pad_sequences(tokenized_n_grams, maxlen=max_padding_len, padding=padding_type)

        return self
    
    def get_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function for separating features and their labels from self.preprocessed_data

        @returns
        X:  numpy.ndarray, features extracted from data
        y:  numpy.ndarray, labels extracted from data
        """

        if self.preprocessed_data is None:
            return "Features and Labels can only be extracted after preprocessing"

        X = self.preprocessed_data[:, :-1]
        y = self.preprocessed_data[:, -1]

        y = one_hot(y, depth=len(self.vocab) + 2)

        return X, y

    def get_vocab(self) -> Set[str]:
        """
        Function to get the vocabulary of text stored in 'data' attribute

        @returns
        vocab: set, set of all words in self.data
        """

        vocab = set()
        for headline in self.data:
            n_grams = self._generate_n_grams(headline)
            try:
                vocab = vocab.union(set(n_grams[-1].split()))
            except IndexError:
                pass
        
        return vocab

    def _generate_n_grams(self, sentence: str) -> List[str]:
        """
        Private method to generate n-gram sequences for given 'sentence'.

        @args
        sentence: str, sentences whose n-grams are to be generated.

        @returns
        n_grams: list, sequence of generated n-grams.
        """

        n_grams = list()
        sentence_words = sentence.split()
        for i in range(2, len(sentence_words) + 1):
            n_grams.append(' '.join(sentence_words[0: i]))

        return n_grams

