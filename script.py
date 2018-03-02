import re, os
from collections import namedtuple
from typing import NamedTuple

import numpy as np
import pandas as pd
import tflearn
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split


def cleaned(content):
    content = content.lower()
    # First remove inline JavaScript/CSS:
    cleaned_content = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", content)
    # Then remove html comments.
    cleaned_content = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned_content)
    # Next remove the remaining tags:
    cleaned_content = re.sub(r"(?s)<.*?>", " ", cleaned_content)
    # Finally deal with whitespace
    cleaned_content = re.sub(r"&nbsp;", " ", cleaned_content)
    cleaned_content = re.sub(r"^$", "", cleaned_content)
    cleaned_content = re.sub("''|,", "", cleaned_content)
    cleaned_content = re.sub(r" {2}", " ", cleaned_content)
    cleaned_content = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", cleaned_content)
    cleaned_content = re.sub(r"\'s", " is", cleaned_content)
    cleaned_content = re.sub(r"\'ve", " have", cleaned_content)
    cleaned_content = re.sub(r"n\'t", " not", cleaned_content)
    cleaned_content = re.sub(r"\'re", " are", cleaned_content)
    cleaned_content = re.sub(r"\'d", " would", cleaned_content)
    cleaned_content = re.sub(r"\'ll", " will", cleaned_content)
    cleaned_content = re.sub(r",", " , ", cleaned_content)
    cleaned_content = re.sub(r"!", " ! ", cleaned_content)
    cleaned_content = re.sub(r"\(", " \( ", cleaned_content)
    cleaned_content = re.sub(r"\)", " \) ", cleaned_content)
    cleaned_content = re.sub(r"\?", " \? ", cleaned_content)
    cleaned_content = re.sub(r"\s{2,}", " ", cleaned_content)
    cleaned_content = re.sub(r"\d+", "", cleaned_content)
    cleaned_content = re.sub(r"[\r\n]+", " ", cleaned_content)
    cleaned_content = re.sub(r'^(https|http)?://.*[\r\n]*', '', cleaned_content)
    return cleaned_content.strip()


def space_tokenizer(w):
    return w.split(' ')


def preprocess(doc):
    return cleaned(doc)


def get_network(vocab_len, n_classes):
    input = tflearn.input_data([None, vocab_len])
    net = tflearn.fully_connected(input, n_classes, weights_init='xavier')
    return tflearn.regression(net, loss='binary_crossentropy', n_classes=n_classes)


DataSet = namedtuple('DataSet', ['x', 'y'])


def load_data():
    # comment_text, labels...
    df = pd.read_csv(os.path.join('data', 'train.csv'))
    tf_idf = TfidfVectorizer(
        stop_words='english',
        tokenizer=space_tokenizer,
        preprocessor=preprocess,
        sublinear_tf=True,
        use_idf=False,
        lowercase=True
    )
    documents = tf_idf.fit_transform(df['comment_text'])
    labels = df.drop(['id', 'comment_text'], axis=1)
    # train_x, test_x, train_y, test_y = train_test_split(documents, labels, test_size=0.3)
    return DataSet(np.array(documents.toarray()), np.array(labels)), len(tf_idf.vocabulary_)


def main():
    print('loading data')
    dataset, vocab_size = load_data()
    nclasses = 6
    net = get_network(vocab_size, nclasses)
    model = tflearn.DNN(net, tensorboard_verbose=1, tensorboard_dir='logs/experiment')
    print('model ready. start fitting the model')
    model.fit(dataset.x, dataset.y,
              n_epoch=10,
              validation_set=0.3,
              show_metric=True,
              batch_size=64,
              shuffle=True,
              run_id='bow_logits'
              )


if __name__ == '__main__':
    main()
