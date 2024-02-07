# adapted code from : https://raw.githubusercontent.com/AssemblyAI-Examples/Machine-Learning-From-Scratch/main/01%20KNN/train.py

import gzip
import numpy as np
from collections import Counter

def compression_distance(x):
    return len(gzip.compress(x.encode()))

def normalized_compression_distance(Cx1, Cx2, Cx1x2):
    return (Cx1x2 - min(Cx1,Cx2)) / max(Cx1, Cx2)

class KNN:
    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        Cx1 = compression_distance(x)

        distances = []
        for x_train in self.X_train:
            Cx2 = compression_distance(x_train)
            x1x2 = " ".join([x, x_train])
            Cx1x2 = compression_distance(x1x2)

            ncd = normalized_compression_distance(Cx1, Cx2, Cx1x2)

            distances.append(ncd)

        # get closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # determine label using majority vote
        most_commun = Counter(k_nearest_labels).most_common(1)[0][0] # return a list of tuple, we just need the most commun labekl val
        return most_commun
