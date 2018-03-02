import re

import numpy as np
import pandas as pd
import tflearn
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer


def cleaned(content):
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
    cleaned_content = re.sub(r"\'s", " \'s", cleaned_content)
    cleaned_content = re.sub(r"\'ve", " \'ve", cleaned_content)
    cleaned_content = re.sub(r"n\'t", " n\'t", cleaned_content)
    cleaned_content = re.sub(r"\'re", " \'re", cleaned_content)
    cleaned_content = re.sub(r"\'d", " \'d", cleaned_content)
    cleaned_content = re.sub(r"\'ll", " \'ll", cleaned_content)
    cleaned_content = re.sub(r",", " , ", cleaned_content)
    cleaned_content = re.sub(r"!", " ! ", cleaned_content)
    cleaned_content = re.sub(r"\(", " \( ", cleaned_content)
    cleaned_content = re.sub(r"\)", " \) ", cleaned_content)
    cleaned_content = re.sub(r"\?", " \? ", cleaned_content)
    cleaned_content = re.sub(r"\s{2,}", " ", cleaned_content)
    cleaned_content = re.sub(r"\d+", "", cleaned_content)
    cleaned_content = re.sub(r"[\r\n]+", " ", cleaned_content)
    return cleaned_content.strip().lower()


def space_tokenizer(w):
    return w.split(' ')


def preprocess(doc):
    return cleaned(doc)


def get_network(vocab_len, n_classes):
    input = tflearn.input_data([vocab_len])
    return tflearn.fully_connected(input, n_classes, weights_init='xavier')


def main():
    # comment_text, labels...
    df = pd.read_csv('data/train.csv')
    tf_idf = TfidfVectorizer(
        stop_words='english',
        tokenizer=space_tokenizer,
        preprocessor=preprocess,
        sublinear_tf=True,
        use_idf=False
    )
    nclasses = 6
    documents = tf_idf.fit_transform(df['comment_text'])
    net = get_network(len(tf_idf.vocabulary_))


if __name__ == '__main__':
    main()
