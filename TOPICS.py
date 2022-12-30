import pandas as pd
import numpy as nm
import matplotlib.pyplot as mtp

df = pd.read_csv('Social_Network_Ads.csv')

x = df.iloc[:,2:4].values
y = df.iloc[:,4].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(x_train,y_train)

ypred = model.predict(x_test)


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

cr = classification_report(y_test, ypred)

ac = accuracy_score(y_test,ypred)

cm = confusion_matrix(y_test,ypred)






# model saving

import pickle

f1 = open(file='naivebayse.pkl',mode="bw")
pickle.dump(model,f1)
f1.close()


f2 = open(file='stndrd.pkl',mode="bw")
pickle.dump(sc,f2)
f2.close()
