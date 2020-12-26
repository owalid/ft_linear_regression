import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def predict(x):
   return slope * x + intercept

def read_csv(file):
  result = []
  data_file = pd.read_csv(file)
  x = data_file.iloc[0:len(data_file),0] # explanatory variable
  y = data_file.iloc[0:len(data_file),1] # reponse variable
  result.append(x)
  result.append(y)
  return np.array(result)


def getSlope(x, y):
  sx = 0
  sy = 0
  sxy = 0
  sum_x = sum(x)
  sum_y = sum(y)
  for i in range(len(x)):
    sx += math.pow(x[i] - sum_x, 2)
    sy += math.pow(y[i] - sum_y, 2)
    sxy += (x[i] - sum_x) * (y[i] - sum_y)

  r = sxy / math.sqrt(sx * sy)
  sx = math.sqrt(sx / len(x) - 1)
  sy = math.sqrt(sy / len(y) - 1)
  return r * sy / sx

def getIntercept(slope, x, y):
  return sum(y) - (slope * sum(x))

data = read_csv('data.csv')
slope = getSlope(data[0], data[1])
intercept = getIntercept(slope, data[0], data[1])
print(predict(23000))

for i in range(len(data[0])):
  plt.scatter(float(data[0][i]),float(data[1][i]))

x = np.linspace(0, 300000, 300000)
y = slope * x + intercept
plt.plot(x, y, '-r')
plt.show()

# y = B0 + B1x
# Slope: B1 = sqrt(Sy / Sx)
# intercept: B0 = sum(y) - B1 * sum(x)

# x = km
# y = price

# Sx = sqrt(sum(x - sum(x)))
# Sy = sqrt(sum(y - sum(y)))

