import numpy as np
from mylinear import MyLinearRegression as MyLR
from utils import read_csv

def get_data_predicted():
  try: 
    return np.load("data_predicted.npz")
  except IOError:
    return False

data_predicted = get_data_predicted()
if data_predicted != False:
  thetas = data_predicted["thetas"]
  cost_history = data_predicted["cost_history"]
  linear_model = MyLR(thetas=thetas, cost_history=cost_history)
  x, y = read_csv('data.csv')
  quited = False
  while not quited:
    number = input("--------------------------------------------------------\nEnter a mileage to get a price estimate \n q or quit to quit the program \n p or plot to plot the charts\n")
    if (number == "q" or number == "Q" or number == "quit"):
      print("Bye 👋")
      quited = True
    elif (number == "p" or number == "P" or number == "plot"):
      linear_model.plot(x, y)
    else:
      arr = np.append(np.array([float(number)]), x)
      arr = linear_model.z_score(arr)
      print("\nThe price for a mileage of ", number, " is: \n")
      print(linear_model.estimate_price(arr[0]), "\n")
else:
  print("No data you need to train model before")