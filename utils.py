import re
from scipy.io.wavfile import read
import numpy as np
from speech_features import features


def user_id(filename):
    match = re.search('usr(\d{4}).*', filename)
    return int(match.group(1)) if match else 0


def process_file(filename):

    # получение отсчётов из файла
    sample_rate, samples = read(filename)

    # вычисление признаков
    feats = features(samples, sample_rate)

    return feats, user_id(filename)


def make_set(files):

    X = np.array([])
    y = np.array([])

    for file in files:

        print(file)

        feats, uid = process_file(file)
        X = feats if len(X) == 0 else np.vstack((X, feats))
        y = np.array([uid]) if len(y) == 0 else np.vstack((y, uid))

    return X, y


def standartize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
