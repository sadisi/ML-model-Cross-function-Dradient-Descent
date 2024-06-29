import numpy as np
import matplotlib.pyplot as plt

#number of data points
n = 5

#data
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 8, 11, 14, 17])

#start with m = 0 and c = 0
m = 0
c = 0

#learning rate
learning_rate = 0.01

for i in range(1, 101):
    y_predict = m * x + c

    #calculate cost
    cost = (1/n) * sum([value**2 for value in (y - y_predict)])

    # plot after each interation cost against m
    plt.scatter(m, cost)

    #calculate gradients
    dm = -(2/n) * sum(x * (y - y_predict))
    dc = -(2/n) * sum(y - y_predict)

    #update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    print("m {}, c {}, cost {}, iteration {}".format(m, c, cost, i))
    plt.show()