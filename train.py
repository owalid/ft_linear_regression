from utils.mylinear import MyLinearRegression as MyLR
import numpy as np
import pandas as pd
from utils.utils import read_csv

# Get data
x, y = read_csv('data.csv') # get informaitons of csv

linear_model = MyLR(np.array([[0.0], [0.0]])) # init theta and new instance of MyLr
print("Start training 🚀")
linear_model.plot(x, y) # plot result
