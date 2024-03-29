import numpy as np
from utils.mylinear import MyLinearRegression as MyLR
from utils.utils import read_csv

def get_data_predicted():
  try: 
    return np.load("data_predicted.npz") # load result fited
  except IOError:
    return False

data_predicted = get_data_predicted()  # get prediction
if data_predicted != False:
  thetas = data_predicted["thetas"]
  cost_history = data_predicted["cost_history"]
  r2_score = data_predicted["r2_score"]
  linear_model = MyLR(thetas=thetas, cost_history=cost_history)
else:
  print("Warning ! You did not train the model \nUsage:\n python train.py")
  linear_model = MyLR(np.array([0, 0]))
  r2_score = 0

try:
  x, y = read_csv('data.csv')
  quited = False
except:
  quited = True
  print("ERROR: can't open file")

try:
  while not quited:
    number = input("--------------------------------------------------------\nEnter a mileage to get a price estimate \n q or quit to quit the program \n p or plot to plot the charts\n r2 or r2_score to display the r2 score\n")
    if (number == "q" or number == "Q" or number == "quit"):
      print("\nBye 👋")
      quited = True
    elif (number == "p" or number == "P" or number == "plot"):
      which_chart = input("\n Wich chart you want to display ? \n c or cost for cost \n r or regression for regression \n b or both for both charts \n")
      if which_chart == "c" or which_chart == "C" or which_chart == "cost":
        linear_model.plot_cost(x, y)
      elif which_chart == "r" or which_chart == "R" or which_chart == "regression":
        linear_model.plot_regression(x, y)
      elif which_chart == "b" or which_chart == "B" or which_chart == "both":
        linear_model.plot_cost(x, y)
        linear_model.plot_regression(x, y)
      else:
        print("\nSorry we can't display:", which_chart, " chart 🙁\n")
    elif (number == "r2" or number == "R2" or number == "r2_score"):
      print("\nThe r2 score is ", r2_score, "\n")
    elif number.isnumeric():
      print("\nThe price for a mileage of ", number, " is: \n")
      arr = np.append(np.array([float(number)]), x)
      arr = linear_model.z_score(arr)
      print(float(linear_model.estimate_price(arr[0])), "$ \n")
    else:
      print("\n can't resolve: ", number, "🙁\n")

except EOFError: # for ctrl + c
  print("\nBye 👋")
  quited = True
except KeyboardInterrupt: # for ctrl + d
  print("\nBye 👋")
  quited = True
