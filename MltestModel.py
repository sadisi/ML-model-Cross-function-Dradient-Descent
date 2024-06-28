import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = pd.read_excel("Book1.xlsx")


plt.scatter(data.videos,data.views, color ="red")
plt.xlabel('Number of Videos')
plt.ylabel('Total Views')

x = np.array(data.videos.values)
y = np.array(data.views.values)


fsModel = LinearRegression()
#we need to convert x axis to 2D but y is same
fsModel.fit(x.reshape((-1,1)),y)

#model Trained
#-------------------------------------------
#new video predect

new_x = np.array([45]).reshape((-1,1))

pred = fsModel.predict(new_x)

plt.scatter(data.videos, data.views, color = 'red')
m, c = np.polyfit(x,y,1)
plt.plot(x,m*x+c)

plt.show()
