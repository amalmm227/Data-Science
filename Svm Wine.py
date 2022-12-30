import pandas as pd
import numpy as nm
import matplotlib.pyplot as mtp

df = pd.read_csv('Wine1.csv')

x = df.iloc[:,0:13].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

xtrain = sc.fit_transform(x_train)
xtest  = sc.transform(x_test)


from sklearn.svm import SVC
model =SVC(kernel='linear',random_state=0)
model.fit(x_train,y_train)
ypred = model.predict(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

ac = accuracy_score(y_test,ypred)

cm = confusion_matrix(y_test,ypred)

cr = classification_report(y_test,ypred)


