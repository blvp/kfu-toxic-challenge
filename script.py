import os
import re
import sys
from collections import namedtuple, Counter
from multiprocessing.pool import Pool

import datetime
import numpy as np
import pandas as pd
import spacy
import tflearn
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer


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
    cleaned_content = re.sub(r"\'s", " 's", cleaned_content)
    cleaned_content = re.sub(r"\'m", " 'm", cleaned_content)
    cleaned_content = re.sub(r"\'ve", " 've", cleaned_content)
    cleaned_content = re.sub(r"n\'t", " n't", cleaned_content)
    cleaned_content = re.sub(r"\'re", " 're", cleaned_content)
    cleaned_content = re.sub(r"\'d", " 'd", cleaned_content)
    cleaned_content = re.sub(r"\'ll", " 'll", cleaned_content)
    cleaned_content = re.sub(r",", " , ", cleaned_content)
    cleaned_content = re.sub(r"!", " ! ", cleaned_content)
    cleaned_content = re.sub(r"\(", " ( ", cleaned_content)
    cleaned_content = re.sub(r"\)", " ) ", cleaned_content)
    cleaned_content = re.sub(r"\?", " ? ", cleaned_content)
    cleaned_content = re.sub(r"\s{2,}", " ", cleaned_content)
    cleaned_content = re.sub(r"\d+", "", cleaned_content)
    cleaned_content = re.sub(r"[\r\n]+", " ", cleaned_content)
    cleaned_content = re.sub(r'^(https|http)?://.*[\r\n]*', '', cleaned_content)
    return cleaned_content.strip()


nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
stops = stopwords.words("english")


def normalize(text, lowercase=True, remove_stopwords=True):
    if lowercase:
        text = text.lower()
    text = nlp(text)
    lemmatized = list()
    for word in text:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops):
                lemmatized.append(lemma)
    return " ".join(lemmatized)


def tokenize(text):
    return wordpunct_tokenize(text)


def cleanup_text(doc):
    return normalize(cleaned(doc))


def get_network(vocab_len, n_classes):
    input = tflearn.input_data([None, vocab_len])
    net = tflearn.fully_connected(input, n_classes, weights_init='xavier')
    return tflearn.regression(net, loss='binary_crossentropy', n_classes=n_classes)


DataSet = namedtuple('DataSet', ['x', 'y'])

num_partitions = 10  # number of partitions to split dataframe
num_cores = 4  # number of cores on your machine


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def cleanup_dataframe(data):
    data['comment_text'] = data['comment_text'].apply(lambda x: cleanup_text(x))
    return data


def tokenize_df(data):
    data['comment_text'] = data['comment_text'].apply(lambda x: tokenize(x))
    return data


def remove_rare_words(df, min_count=4):
    df = parallelize_dataframe(df, tokenize_df)
    docs = df['comment_text']
    word_cnt = Counter([w for doc in docs for w in doc])
    df['comment_text'] = df['comment_text'].apply(lambda doc: ' '.join([w for w in doc if word_cnt[w] >= min_count]))
    return df


def load_data(filename):
    # comment_text, labels...
    df = pd.read_csv(os.path.join('data', filename))
    df = parallelize_dataframe(df, cleanup_dataframe)
    print('cleaned text data')
    return df


def batchify(x, y, batch_size=64):
    nsamples = x.shape[0]
    index = 0
    for offset in range(0, nsamples, batch_size):
        yield index, x[offset: batch_size + offset], y[offset: batch_size + offset]
        index += 1


def main():
    print('loading data')
    train_df = load_data('train.csv')
    tf_idf = TfidfVectorizer(
        stop_words='english',
        tokenizer=tokenize,
        preprocessor=None,
        sublinear_tf=True,
        use_idf=False,
        lowercase=True,
        min_df=4
    )
    documents = tf_idf.fit_transform(train_df['comment_text'])
    labels = train_df.drop(['id', 'comment_text'], axis=1)
    train_x, test_x, train_y, test_y = train_test_split(documents, labels, test_size=0.3)
    test_x = np.array(test_x.toarray())
    vocab_size = len(tf_idf.vocabulary_)
    batch_size = 64
    learning_rate = 0.01
    epochs = 10
    nclasses = 6
    graph = tf.Graph()
    with graph.as_default():
        initializer = xavier_initializer(dtype=tf.float32)
        x = tf.placeholder(tf.float32, [None, vocab_size], 'input')
        y = tf.placeholder(tf.float32, [None, nclasses], 'input_labels')
        h1 = tf.Variable(initializer([vocab_size, nclasses]), name='hidden1')
        b1 = tf.Variable(tf.random_uniform([nclasses]), name='bias1')
        y_pred = x * h1 + b1
        loss = tf.losses.sigmoid_cross_entropy(y, y_pred, label_smoothing=1)
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

        with tf.Session(graph).as_default() as sess:
            merged = tf.summary.merge_all()
            batch_writer = tf.summary.FileWriter('./logs/batch')
            epoch_writer = tf.summary.FileWriter('./logs/epoch')
            tf.initialize_all_variables().run()
            total = train_x.shape[0] / batch_size
            for epoch in range(epochs):
                # shuffle(self.batches)
                print("Batches shuffled")
                print("-----------------")
                sys.stdout.flush()
                accumulated_loss = 0
                num_batches = total / batch_size
                for batch_index, input_x, input_y in batchify(train_x, train_y, batch_size=batch_size):
                    feed_dict = {
                        x: np.array(input_x),
                        y: np.array(input_y)
                    }
                    summaries, _, total_loss_, = sess.run(
                        [merged, optimizer, loss], feed_dict=feed_dict
                    )
                    accumulated_loss += total_loss_
                    print("epoch: {0}/{1}".format(epoch + 1, epochs))
                    print("batch: {0}/{1}".format(batch_index + 1, num_batches))
                    print("average loss: {}".format(accumulated_loss / 100))
                    print("-----------------")
                    batch_writer.add_summary(summaries, epoch * num_batches + batch_index)
                    sys.stdout.flush()
                    accumulated_loss = 0

                summaries, epoch_test_loss, = sess.run(
                    [merged, loss], feed_dict={
                        x: test_x,
                        y: test_y
                    }
                )
                epoch_writer.add_summary(summaries, epoch)
                print("epoch: {0}/{1}".format(epoch + 1, epochs))
                print("val loss: {}".format(epoch_test_loss))
                print("-----------------")
                sys.stdout.flush()
                print("Epoch finished: {}".format(datetime.datetime.now().time()))
                print("=================")


if __name__ == '__main__':
    main()
