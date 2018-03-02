import os
import re
from collections import namedtuple, Counter
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
import spacy
import tflearn
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


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
    df = remove_rare_words(df, min_count=4)
    print('removed rare words')
    return df

def main():
    print('loading data')
    train_df = load_data('train.csv')
    tf_idf = TfidfVectorizer(
        stop_words='english',
        tokenizer=tokenize,
        preprocessor=None,
        sublinear_tf=True,
        use_idf=False,
        lowercase=True
    )
    documents = tf_idf.fit_transform(train_df['comment_text'])
    labels = train_df.drop(['id', 'comment_text'], axis=1)
    # train_x, test_x, train_y, test_y = train_test_split(documents, labels, test_size=0.3)
    dataset = DataSet(np.array(documents.toarray()), np.array(labels))
    vocab_size = len(tf_idf.vocabulary_)

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
