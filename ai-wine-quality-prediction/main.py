import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv('winequality-red.csv', sep=';')[:1000]
keys = data.keys()
keys = [i for i in keys]
predict = "quality"

z = np.array(data.drop(columns=predict))
y = np.array(data[predict])
best = 0
for i in range(1000):
    z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.1)
    linear = LinearRegression()
    linear.fit(z_train, y_train)
    acc = linear.score(z_test, y_test)
    if acc > best:
        best = acc
        print(best)
        with open('winemodel.pickle', 'wb') as f:
            pickle.dump(linear, f)

pickle_in = open('winemodel.pickle', 'rb')
linear = pickle.load(pickle_in)
data = pd.read_csv('winequality-red.csv', sep=';')[1000:]
d = pd.read_csv('winequality-red.csv', sep=';')[1000:]
print(d)
predictions = linear.predict(data.drop(columns=predict))
d = d.reset_index(drop=True)
for i in range(len(predictions) - 1):
    print(predictions[i], d['quality'][i])

key = keys[3]
style.use('ggplot')
pyplot.scatter(data[key], data['quality'])
pyplot.xlabel(key)
pyplot.ylabel('quality')
pyplot.show()
