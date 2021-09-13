import numpy as np
import math
import matplotlib.pyplot as plt

class MyLinearRegression():
  """
	Description:
		My personnal linear regression class to fit like a boss.
	"""
  def __init__(self, thetas, cost_history=None, alpha=0.01, max_iter=600):
    self.alpha = alpha
    self.max_iter = max_iter
    self.thetas = thetas.astype(np.float32)
    self.cost_history = cost_history if cost_history is not None else np.zeros(self.max_iter)

# PRIVATE METHODS
  def __mean_(self, x): # function used to calculate mean/average
    len_x = len(x)
    result = 0
    for i in range(len_x):
      result += float(x[i])
    return float(result / len_x)

  def __var_(self, x): # function used to calculate variance
    mean = self.__mean_(x)
    len_x = len(x)
    result = 0
    for i in range(len_x):
      result += (x[i] - mean)**2
    return float(result / len_x)

  def __std_(self, x): # function used to calculate standard deviation
    return math.sqrt(self.__var_(x))

  def __add_intercept(self, x):
    return np.c_[np.ones((x.shape[0], 1)), x]
    
  def __gradient(self, x, y): # function that applies the descending gradient on theta
    x_pr = self.__add_intercept(x)
    m = y.shape[0]
    return (1 / m) * np.dot(np.transpose(x_pr), (np.dot(x_pr, self.thetas) - y))

  def __r2score_(self, y, y_hat):
    y_mean = self.__mean_(y)
    return 1 - ((y_hat - y)**2).sum() / ((y - y_mean)**2).sum()

  def __mse_(self, y, y_hat): # function used to calculate mean squared error
    y,  y_hat = np.array(y), np.array(y_hat)
    return np.square(np.subtract(y,y_hat)).mean()
# END PRIVATE METHODS

# PUBLIC METHODS
  def z_score(self, x):
    mean = self.__mean_(x)
    std = self.__std_(x)
    return (x - mean) / std

  def fit_(self, x, y):
    for i in range(self.max_iter):
      self.thetas -= self.alpha * self.__gradient(x, y)
      self.cost_history[i] = self.__mse_(y, self.predict_(x))
    return self.thetas, self.cost_history
  
  def predict_(self, x):
    return np.dot(self.__add_intercept(x), self.thetas)

  def plotCost(self, x, y): # plot cost function
    plt.plot(range(self.max_iter), self.cost_history, 'r')
    plt.show()
  
  def plotRegression(self, x, y): # plot regression
    x_normalized = self.z_score(x)
    plt.scatter(x, y, c='g')
    plt.plot(x, self.predict_(x_normalized), 'r')
    plt.show()

  def plot(self, x, y):
    x_normalized = self.z_score(x)
    self.fit_(x_normalized, y)
    r2_score = self.__r2score_(y, self.predict_(x_normalized))
    print("The r2 score is ", r2_score, "\n")
    np.savez("data_predicted", thetas=self.thetas, cost_history=self.cost_history, r2_score=r2_score) # saves data in npz file

    # plot cost and regression
    self.plotCost(x, y)
    self.plotRegression(x, y)

  def estimate_price(self, X):
    return self.thetas[0] + (self.thetas[1] * X)