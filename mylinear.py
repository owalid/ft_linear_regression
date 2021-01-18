import os
import numpy as np
import math
import matplotlib.pyplot as plt

class MyLinearRegression():
  """
	Description:
		My personnal linear regression class to fit like a boss.
	"""
  def __init__(self,  thetas, alpha=0.01, max_iter=1000):
    self.alpha = alpha
    self.max_iter = max_iter
    self.thetas = thetas.astype(np.float32)

  @staticmethod
  def mean_(x):
    len_x = len(x)
    result = 0
    for i in range(len_x):
      result += float(x[i])
    return float(result / len_x)

  @staticmethod
  def var_(self, x):
    mean = self.mean_(x)
    len_x = len(x)
    result = 0
    for i in range(len_x):
      result += (x[i] - mean)**2
    return float(result / len_x)

  @staticmethod
  def std_(self, x):
    return math.sqrt(self.var_(self, x))

  def z_score(self, x):
    mean = self.mean_(x)
    # print(self.std_(self, x))
    std = self.std_(self, x)
    return (x - mean) / std

  @staticmethod
  def add_intercept(x):
    return np.c_[np.ones((x.shape[0], 1)), x]
    
  @staticmethod
  def gradient(self, x, y):
    x_pr = self.add_intercept(x)
    m = y.shape[0]
    return (1 / m) * np.dot(np.transpose(x_pr), (np.dot(x_pr, self.thetas) - y))

  # @staticmethod
  def mse_(self, y, y_hat): 
    y,  y_hat = np.array(y), np.array(y_hat)
    return np.square(np.subtract(y,y_hat)).mean() 

  def fit_(self, x, y):
    for _ in range(self.max_iter):
      self.thetas -= self.alpha * self.gradient(self, x, y)
    return self.thetas
  
  def predict_(self, x):
    return np.dot(self.add_intercept(x), self.thetas)

  def cost_elem_(self, x, y):
    m = y.shape[0]
    y_hat = self.predict_(x)
    return (1 / (2 *m)) * (y_hat - y)**2
  
  def cost_(self, x, y):
    return np.sum(self.cost_elem_(x, y))

  def plot(self, x, y):
    normalize_data = self.z_score(x)
    self.fit_(normalize_data, y)
    plt.scatter(x, y, c='g')
    plt.plot(x, self.predict_(normalize_data), 'r')
    plt.show()

  def write(self):
    try:
      os.remove("theta.txt")
    except IOError:
      print("no file to remove")
    result = str(self.thetas[0][0])+":"+str(self.thetas[1][0])
    f = open("theta.txt", "a+")
    f.write(result)
    f.close()

  def estimate_price(self, X):
    return self.thetas[0] + (self.thetas[1] * X)