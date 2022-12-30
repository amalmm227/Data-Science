import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')

x = df.iloc[:,3:].values
plt.scatter(x[:,0],x[:,1],marker='+')
plt.show()

#Applying Kmeans  Clustering

from sklearn.cluster import KMeans
model = KMeans()

WCSS = []
for i in range(1,11):
    model = KMeans(n_clusters=i)
    model.fit(x)
    a=model.inertia_
    WCSS.append(a)
    
plt.plot(range(1,11),WCSS)
plt.xlabel('no of clusters')
plt.ylabel('Wcss') 
plt.title('Finding Clusters') 
plt.show()
model1 = KMeans(n_clusters=5)
model1.fit(x)
y=model1.predict(x)  
    
print(set(y))


#Visualization of clusterd Data

plt.scatter(x[y==0,0],x[y==0,1],color='r',label='First Cluster')
plt.scatter(x[y==1,0],x[y==1,1],color='g',label='Second Cluster')
plt.scatter(x[y==2,0],x[y==2,1],color='b',label='Third Cluster')
plt.scatter(x[y==3,0],x[y==3,1],color='yellow',label='Fourth Cluster')
plt.scatter(x[y==4,0],x[y==4,1],color='k',label='Fifth Cluster')
plt.legend()



#To find Centeroid


plt.scatter(model1.cluster_centers_[:,0],model1.cluster_centers_[:,1],c='orange',s=200)
plt.show()
