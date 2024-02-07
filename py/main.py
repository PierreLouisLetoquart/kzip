import gzip
import pandas as pd
import numpy as np
from collections import Counter

data = pd.read_csv("/the/path/to/data/sentimentdataset.csv")

if not data.columns.isin(['Text', 'Sentiment']).any().all():
    print("Error: 'Text' and/or 'Sentiment' are not present in your DataFrame")

X = data['Text']
y = data['Sentiment']

data_len = len(data)
train_size = int(data_len*0.8)

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[data_len-train_size:]
y_test = y[data_len-train_size:]

predictions = []
for x1 in X_test:
    Cx1 = len(gzip.compress(x1.encode()))
    distances = []
    for x2 in X_train:
        Cx2 = len(gzip.compress(x2.encode()))
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1,Cx2)) / max(Cx1, Cx2)
        distances.append(ncd)

    k_indices = np.argsort(distances)[:3]
    k_nearest_labels = [y_train[i] for i in k_indices]

    most_commun = Counter(k_nearest_labels).most_common(1)[0][0]
    predictions.append(most_commun)

# print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
