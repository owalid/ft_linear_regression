import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
plt.rcParams['figure.figsize'] = (12.0, 9.0)

def normalize_data(data):
  min_data = data.min()
  max_data = data.max()
  result = (data - min_data) / (max_data - min_data)
  return result

def estimate_price(theta, X):
  return theta[0] + (theta[1] * X)

def gradient_descent(X, y, theta, learning_rate, iterations):
  inv_m = 1 / len(y)

  for _ in range(iterations):
    tmp_theta_0 = learning_rate * inv_m * sum(estimate_price(theta, X) - y)
    tmp_theta_1 = learning_rate * inv_m * sum(X * (estimate_price(theta, X) - y))
    theta[0] -= tmp_theta_0
    theta[1] -= tmp_theta_1
  
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
x = normalize_data(data[0])
y = normalize_data(data[1])

# matix X
# X = np.hstack((x, np.ones(x.shape)))

theta = [0, 0]
final_theta = gradient_descent(x, y, theta, 0.001, 10000)
print(final_theta)
print(estimate_price(final_theta, 60000))
# plt.scatter(x, y)
# plt.plot(x, estimate_price(final_theta, x), c='r')
# plt.show()
