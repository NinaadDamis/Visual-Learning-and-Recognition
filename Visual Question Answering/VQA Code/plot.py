from matplotlib import pyplot as plt
import numpy as np
import csv

x = []
y = []
  
with open('x_transformer.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    i = 0
    for row in plots:
        i+=1
        print(int(float(row[0])))
        x.append(int(float(row[0])))
        # if(i == 910) : break
    print("len = ", len(x))
    print("type = ", type(x[0]))

with open('y_transformer.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    i = 0
    for row in plots:
        i+=1
        print(int(float(row[0])))
        y.append(int(float(row[0])))
        # if(i == 910) : break

    print("len = ", len(x))
    print("type = ", type(x[0]))
  
plt.bar(x, y, color = 'g', width = 0.72)
plt.ylabel("Frequencies")
plt.xlabel("X axis")
plt.show()