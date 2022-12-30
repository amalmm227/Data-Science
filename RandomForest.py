import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt

df = pd.read_csv('Social_Network_Ads.csv')
x = df.drop(['User ID','Gender','Purchased'],axis=1)
y =df['Purchased']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10,criterion='entropy')

model.fit(x_train,y_train)

ypred = model.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

ac = accuracy_score(y_test,ypred)

cm = confusion_matrix(y_test,ypred)

cr = classification_report(y_test,ypred)

from sklearn.metrics import roc_curve,auc,roc_auc_score
fpr,tpr,thresh=roc_curve(y_test,ypred)


a=auc(fpr,tpr)
plt.plot(fpr,tpr,color="green",label=("AUC value: %0.2f"%(a)))
plt.plot([0,1],[0,1],"--",color="red")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC-AUC CURVE")
plt.legend(loc="best")
plt.show()


