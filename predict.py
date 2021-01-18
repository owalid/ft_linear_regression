import numpy as np
from mylinear import MyLinearRegression as MyLR
from utils import read_csv

def minmax(x):
  min_x = min(x)
  max_x = max(x)
  return (x - min_x) / (max_x - min_x)

def getTheta():
  try: 
    f = open("theta.txt", "r")
    contents = f.read()
    contents = contents.split(":")
    result = np.array([float(contents[0]), float(contents[1])])
    return result
  except IOError:
    return False

theta = getTheta()
if theta.any():
  linear_model = MyLR(theta)
  x, y = read_csv('data.csv')
  arr = np.append(np.array([220000]), x)
  arr = linear_model.z_score(arr)
  print(linear_model.estimate_price(arr[0]))
else:
  print("No data you need to train model before")