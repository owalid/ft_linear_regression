from mylinear import MyLinearRegression as MyLR
import numpy as np
import pandas as pd

def read_csv(file):
  data = pd.read_csv(file)
  x = np.array(data["km"]).reshape(-1,1)
  y = np.array(data["price"]).reshape(-1,1)
  return x, y

# Get data
x, y = read_csv('data.csv')

linear_model = MyLR(np.array([[0.0], [0.0]]))
# y_model = linear_model.predict_(x)
linear_model.plot(x, y)
linear_model.write()
