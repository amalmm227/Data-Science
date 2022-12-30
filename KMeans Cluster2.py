import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs

x,y = make_blobs(n_samples=400,centers=4,cluster_std=.6,random_state=0)

plt.scatter(x[:,0],x[:,1])
plt.show()

model = KMeans(n_clusters=4)

model.fit(x)

ypred = model.predict(x)


#VisualizATION 

plt.scatter(x[:,0],x[:,1],c=ypred,s=20,cmap='summer')

centers = model.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],color='k',s=50)
plt.show()
