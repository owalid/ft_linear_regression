import pandas as pd
import numpy as np

def read_csv(file):
  data = pd.read_csv(file)
  x = np.array(data["km"]).reshape(-1,1)
  y = np.array(data["price"]).reshape(-1,1)
  return x, y