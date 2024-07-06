import numpy as np
np.random.seed(1)

def step_function(x):
  return np.where(x >= 0, 1, 0)

class Perceptron:

  def __init__(self, eta=0.01, epochs=10) -> None:
    self._eta = eta
    self._epochs = epochs

  def predict(self, X):
    z = np.dot(X, self.weights) + self.bias
    return step_function(z)

  def fit(self, X, y):
    self.weights = 2 * np.random.random((X.shape[1], 1)) - 1
    self.bias = 0
    for _ in range(self._epochs):
      predictions = self.predict(X)
      loss = y - predictions
      self.weights += self._eta * np.dot(X.T, loss)
      self.bias += self._eta * np.mean(loss)
