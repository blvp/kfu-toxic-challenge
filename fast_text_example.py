import fastText
import numpy as np
from sklearn.metrics import auc, roc_auc_score

from script import load_data


def to_fast_text_format(labels):
    def _to_fast_text_format(item):
        prepended = ["__label__{}".format(label) for label in labels if item[label] == 1]
        if len(prepended) == 0:
            prepended.append('__label__none')
        return " ".join(prepended) + " " + item.comment_text

    return _to_fast_text_format


# df = load_data('train.csv')

labels = [
    '__label__none',
    '__label__toxic',
    '__label__obscene',
    '__label__insult',
    '__label__identity_hate',
    '__label__severe_toxic',
    '__label__threat'
]

# out = df.apply(to_fast_text_format(labels), axis=1)

ft = fastText.train_supervised('data/fast_text_train', lr=0.1, saveOutput=1)
ft.save_model('model.bin')


def eval(input_file, ft, name):
    with open(input_file, 'r') as f:
        ys = []
        y_preds = []
        for line in f.readlines():
            sent = line.split()
            sample_labels, sample = sent[:7], sent[7:]
            sample = " ".join(sample)
            y = [1 if label in sample_labels else 0 for label in labels]
            ys.append(y)
            _, pred = ft.predict(sample, k=7)
            y_preds.append(pred)
        y_preds = np.array(y_preds)
        ys = np.array(ys, dtype=np.float32)
        y_preds = y_preds[:, 1:]
        ys = ys[:, 1:]
        print(name, "auc: ", roc_auc_score(ys, y_preds, average="micro"))


eval('data/fast_text_test', ft, 'test')
eval('data/fast_text_train', ft, 'train')
