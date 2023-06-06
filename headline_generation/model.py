import numpy as np

from datetime import datetime
from typing import List

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class HeadlineGenerator:

    def __init__(self):
        self.model = None
        self.model_train_history = None

    def create(self, layers: List, loss_fn: str, optimizer: str) -> None:
        """
        Method to create Headline Generator model

        @args
        layers:     list, layers passed to tensorflow's Sequential model
        loss_fn:    str, loss function used for model training
        optimizer   str, optimizer used for model training
        """
        self.model = Sequential(layers)
        self.model.compile(optimizer=optimizer, loss=loss_fn)

    def train(self, X, y, epochs) -> None:
        """
        Method for training the model and saving it

        @args
        X:      np.ndarray, training features
        y:      np.ndarray, training labels
        epochs: int, numper of epochs for training
        """
        self.model_train_history = self.model.fit(X, y, epochs=epochs)
        current_time = datetime.now().strftime("%H:%M:%S")
        self.model.save(f'headline_generator-{current_time}.h5')

    def load_saved(self, model_path) -> None:
        """
        Method for loading an alreadytrained headline generator model

        @args
        model_path: str, path of the saved model file
        """
        self.model =  load_model(model_path)
        self.model_train_history = None

    def generate_headline(self, max_sentence_length, sequence_len, tokenizer, word_sample_size, initial_sentence=None):
        """
        Method for generating headlines from headline generator model
        
        @args
        max_sentence_length: int, the maximum number of words expected in generated headline
        sequence_len:        int, sequence length used in preprocessing data
        tokenizer:           tensorflow.keras.preprocessing.text.Tokenizer,
                             tokeninzer used for preprocessing data
        word_sample_size:    int, number of top predictions to consider while sampling precdicted word
        initial_sentence:    str or None, sentence to start generating headline from
        """

        if initial_sentence is None:
            initial_sentence = '<START>'

        generated_sentence = initial_sentence.split()
        for _ in range (max_sentence_length-2):

            generated_sentence_tokens = tokenizer.texts_to_sequences([generated_sentence])
            generated_sentence_padded_tokens = pad_sequences(generated_sentence_tokens, maxlen=sequence_len)

            pred_tokens = self.model.predict(generated_sentence_padded_tokens)[0]
            top_n_pred_tokens = np.argpartition(pred_tokens, -word_sample_size)[-word_sample_size:]
            pred_token = np.random.choice(top_n_pred_tokens, size=1)

            pred_text = tokenizer.sequences_to_texts([pred_token])[0]
            generated_sentence.append(pred_text)
            if pred_text == '<END>':
                return ' '.join(generated_sentence)

        generated_sentence.append('<END>')
        generated_sentence = ' '.join(generated_sentence)
        return generated_sentence
