import numpy as np
from mylinear import MyLinearRegression as MyLR

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
if (theta != False):
  linear_model = MyLR(theta)
  print(linear_model.estimate_price(82029))
else:
  print("No data you need to train model before")