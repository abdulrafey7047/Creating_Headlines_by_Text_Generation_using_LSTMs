
# Headline Generation Using LSTMs

In this project a basic `LSTM-DNN` model is trained on [New York Times Comments](https://www.kaggle.com/datasets/aashita/nyt-comments) dataset to generate headlines. Although the dataset has a lot of features but we only use the `healdine` column to train our model.

## Tools Used
- [Tensorflow](https://www.tensorflow.org/) - For Deep Learning libraries
- [Keras](https://keras.io/) - To create the model
- [Pandas](https://pandas.pydata.org/) and [Numpy](https://numpy.org/) - To pre-process the data

## Pre-Processing Techniques
 - Removing punctuations.
 - Using `<START>`, `<END>` and `<OOV>` tokens.
 - Generating `n-grams`.
 - `Tokenizing` and `Padding` all sequences of `n-grams`.
 - Using Stanford's 50-dimensional [GloVe](https://nlp.stanford.edu/projects/glove/) `Embeddings` which are trained on 6 billion tokens.

## Usage
To use the project you should `python` installed and `pip` configured. If you don't already go through these to [install python](https://realpython.com/installing-python/) and [configure pip](https://realpython.com/what-is-pip/). After the inital setup you can install all dependencies using
```
pip install -r requirements.txt
```
Following is a documentation of methods in `DataPreProcessor` and `HeadlineGenerator` classes.

### DataPreProcessor
- `clean_data()`
Method to clean data by converting text to lower case and removing punctuations.

- `preprocess()`
Method to preprocess data and bring it into a format that can be fed to the `HeadlineGenerator` class

|  Argument Name      | Type    | Description                |
| :--------           | :------ | :------------------------- |
| `oov_token`         | `string`| out of vocabulary token, passed to tensorflow's tokenizer. |
| `max_padding_len`   | `int`   | length to which all sequences will be padded to. |
| `padding_type`      | `string`| where to add padding tokens, either one from `pre` or `post`. |
| `tokenizing_filters`| `list`  | filters passed to tensorflow's tokenizer, no filters are passed by default. |

- `get_features_and_labels()`
Method for separating features and their labels from self.preprocessed_data. It returns features and labels extracted from data in a `tuple` of format `(features, labels)`

- `get_vocab()`
Method to get the vocabulary of text stored in `data` attribute
returns the set of all words in `self.data`

### HeadlineGenerator
- `create()`
Method to create Headline Generator model.

|  Argument Name      | Type    | Description                |
| :--------           | :------ | :------------------------- |
| `layers`            | `list`  | layers passed to `tensorflow`'s `Sequential` model. | 
| `optimizer`         | `string`| `loss function` used for model training by `tensorflow`. | 
| `tokenizing_filters`| `string`| `optimizer` used for model training by `tensorflow`. |

- `train()`
Method for training the model and saving it.

|  Argument Name | Type         | Description                   |
| :--------      | :------      | :-------------------------    |
| `X`            | `np.ndarray` | training features             | 
| `y`            | `np.ndarray` | training labels               | 
| `epochs`       | `int`        | numper of epochs for training |

- `load_saved()`
Method for loading an already trained headline generator model.

|  Argument Name | Type    | Description                    |
| :--------      | :-------| :-------------------------     |
| `model_path`   | `string`| path of the saved model file   | 

- `generate_headline()`
Method for generating headlines from headline generator model. Returns a string of generated headline.

|  Argument Name        | Type     | Description                |
| :--------             | :------  | :------------------------- |
| `max_sentence_length` | `int`    | the maximum number of words expected in generated headline. | 
| `sequence_len`        | `int`    | sequence length used in preprocessing data. | 
| `tokenizer`           | `tensorflow.keras.preprocessing.text.Tokenizer`| tokeninzer used for preprocessing data. |
| `word_sample_size`    | `int`    | number of top predictions to consider while sampling precdicted word. |
| `initial_sentence`    | `string` or `None` | sentence to start generating headline from |

### Utility methods
- `generate_embedding_matrix_from_file()`
Function to generate a numpy matrix of GloVe embeddings given the embedding file path.

**NOTE:** This function only parsees the format of GloVe embeddings, it will not work on any other format.

|  Argument Name     | Type     | Description                           |
| :--------          | :------  | :-------------------------            |
| `filepath`         | `string` | path to embedding file.               | 
| `vocab_size`       | `int`    | number of words in vocabulary.        | 
| `embedding_dim`    | `int`    | dimesion for embedding of each word.  |
| `token_word_index` | `dict`   | map of word to their `int` tokens.    | 



Usage of all these classes and methods can be found in [sample.ipynb](https://github.com/abdulrafey7047/Creating_Headlines_by_Text_Generation_using_LSTMs/blob/main/sample.ipynb)

If you want to reach me reagrding this repository, feel free to contact me on my [linkedin](https://www.linkedin.com/in/abdulrafey183/)

Happy Coding!
