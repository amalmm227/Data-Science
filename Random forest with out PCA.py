import pandas as pd

df = pd.read_csv('Wine1.csv')
df.corr()['Customer_Segment'].sort_values()

x = df.drop(['Customer_Segment'],axis=1)

y = df['Customer_Segment']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
 
X=sc.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10,random_state=0,criterion='entropy')

model.fit(x_train,y_train)

ypred = model.predict(x_test)


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

ac = accuracy_score(y_test,ypred)

cm = confusion_matrix(y_test, ypred)

cr = classification_report(y_test,ypred)
