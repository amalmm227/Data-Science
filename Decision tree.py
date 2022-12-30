import pandas as pd

df = pd.read_csv('Social_Network_Ads.csv')

x = df.drop(['User ID','Gender','Purchased'],axis=1)
y =df['Purchased']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion="entropy",random_state=0)
model.fit(x_train,y_train)
ypred = model.predict(x_test)

from sklearn.tree import export_text

tree = export_text(model,feature_names=["Age","Salary"])

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

ac1 = accuracy_score(y_test,ypred)

cm1 = confusion_matrix(y_test, ypred)

cr1 = classification_report(y_test,ypred)


#ROC AOC CURVE

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train,y_train)
ypred1 = model1.predict(x_test)





from sklearn.metrics import roc_auc_score,roc_curve,auc
fpr,tpr,thresh=roc_curve(y_test,ypred)
a = auc(fpr,tpr)


fpr1,tpr1,thresh = roc_curve(y_test,ypred1)
b = auc(fpr1,tpr1)

plt.plot(fpr,tpr,color="green",label=("AUC value of Decision tree: %0.2f"%(a)))
plt.plot(fpr1,tpr1,color="blue",label=("AUC value of logistic Regression: %0.2f"%(b)))
plt.plot([0,1],[0,1],"--",color="red")
plt.xlabel("False positive rate")
plt.ylabel("True Positive rate")
plt.title("ROC-AUC Curve")
plt.legend(loc="best")
plt.show()



