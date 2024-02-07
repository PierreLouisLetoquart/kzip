import pandas as pd
import numpy as np
from KNN import KNN

data = pd.read_csv("/Users/pilou/Programs/kzip/data/sentimentdataset.csv")

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

classifier = KNN(k=3)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
