import numpy as np
import pandas as pd

def read_csv(file):
  data = pd.read_csv(file)
  x = np.array(data["km"]).reshape(-1,1) # get kms
  y = np.array(data["price"]).reshape(-1,1) # get prices
  return x, y