import pandas as pd
import numpy as nm
import matplotlib.pyplot as mtp

df = pd.read_csv('Social_Network_Ads.csv')

x = df.iloc[:,2:4].values
y = df.iloc[:,-1].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=4,metric='minkowski',p=2)
acc=[]
model.fit(x_train,y_train)

ypred = model.predict(x_test)

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

cm = confusion_matrix(y_test,ypred)

ac = accuracy_score(y_test,ypred)

cr = classification_report(y_test, ypred)

from matplotlib.colors import ListedColormap  
x_set, y_set = x_train, y_train  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2, model.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('SVM (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Purchased')  
mtp.legend()  
mtp.show()


