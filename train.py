import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
plt.rcParams['figure.figsize'] = (12.0, 9.0)

def estimate_price(theta, X):
  return theta[0] + (theta[1] * X)

def gradient_descent(X, y, theta, learning_rate, iterations):
  m = len(y)

  i = 0
  while i < iterations:
    theta[0] -= learning_rate * (1 / m) * sum(estimate_price(theta, X) - y)
    theta[1] -= learning_rate * (1 / m) * sum(X * (estimate_price(theta, X) - y))
    i += 1
  
  return theta

def read_csv(file):
  result = []
  data_file = pd.read_csv(file)
  x = data_file.iloc[0:len(data_file),0] # explanatory variable
  y = data_file.iloc[0:len(data_file),1] # reponse variable
  result.append(x)
  result.append(y)
  return np.array(result)

# Get data
data = read_csv('data.csv')

# convert array to vector
x = data[0]
y = data[1]

# matix X
# X = np.hstack((x, np.ones(x.shape)))

theta = [0, 0]
final_theta = gradient_descent(x, y, theta, 0.001, 1000)
print(final_theta)

plt.scatter(x, y)
plt.plot(x, estimate_price(final_theta, x), c='r')
# plt.show()
