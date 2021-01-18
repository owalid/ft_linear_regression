from mylinear import MyLinearRegression as MyLR
from utils import read_csv
import numpy as np

# Get data
x, y = read_csv('data.csv')

linear_model = MyLR(np.array([[0.0], [0.0]]))
# y_model = linear_model.predict_(x)
linear_model.plot(x, y)
linear_model.write()
