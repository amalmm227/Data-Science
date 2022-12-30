import pandas as pd

df = pd.read_csv('Wine1.csv')
df.corr()['Customer_Segment'].sort_values()

x = df.drop(['Customer_Segment'],axis=1)

y = df['Customer_Segment']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X=sc.fit_transform(x)

#Applying PCA on features #Principle Componentn Analysis#for Diamensionalty reduction


from sklearn.decomposition import PCA 

pca = PCA(n_components=2)

x1 = pca.fit_transform(X)


explained_variance =pca.explained_variance_ratio_

print(sum(explained_variance))


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x1,y,test_size=0.2,random_state=0)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10)

model.fit(x_train,y_train)

ypred = model.predict(x_test)


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

ac = accuracy_score(y_test,ypred)

cm = confusion_matrix(y_test, ypred)

cr = classification_report(y_test,ypred)
