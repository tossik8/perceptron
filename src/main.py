import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from perceptron import Perceptron

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Iris.csv'))
print(df.describe())

colours = df['Species'].map({'Iris-setosa': 'r',
                             'Iris-versicolor': 'g',
                             'Iris-virginica': 'b'})
x_label, y_label = 'SepalLengthCm', 'SepalWidthCm'
plt.scatter(df[x_label], df[y_label], c=colours)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title('Iris')
plt.show()

X = df.drop(columns=['Id', 'Species'])
df.loc[df['Species'] != 'Iris-setosa', 'Species'] = 0
df.loc[df['Species'] == 'Iris-setosa', 'Species'] = 1
y = df['Species'].values.reshape(150, 1).astype(float)

perceptron = Perceptron(epochs=25)
perceptron.fit(X.values[:, 0:2], y)

predictions = perceptron.predict(X.values[:, 0:2])
print(f'Accuracy: {accuracy_score(y, predictions)}')

x2 = -(perceptron.weights[0] / perceptron.weights[1]) * df[x_label] - (perceptron.bias / perceptron.weights[1])
plt.scatter(df[x_label], df[y_label], c=colours)
plt.plot(df[x_label], x2, c='y')
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title('Decision Boundary for Perceptron')
plt.show()
