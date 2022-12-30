import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits


digits = load_digits()
digits.data.shape

model = KMeans(n_clusters=10,random_state=0)
model.fit(digits.data)
ypred = model.predict(digits.data)

model.cluster_centers_.shape

fig,ax = plt.subplots(2,5,figsize=(8,3))

centers = model.cluster_centers_.reshape(10,8,8)

for i,center in zip(ax.flat,centers):
    i.set(xticks=[],yticks=[])
    i.imshow(center,interpolation='nearest',cmap=plt.cm.binary)