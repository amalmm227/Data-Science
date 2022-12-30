import pandas as pd
import numpy as nm
import  matplotlib.pyplot as mtp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV
df = pd.read_csv('Social_Network_Ads.csv')
df
x = df.iloc[:,2:4].values


y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size =0.8,random_state=0)
 
sc =StandardScaler()
xtrain = sc.fit_transform(x_train)
xtest = sc.transform(x_test)
 
 
from sklearn.svm import SVC
#model = SVC(kernel='linear',random_state=0)

params = {'kernel':['linear','poly','rbf','sigmoid','precomputed'],'gamma':[5,10,15],'C':[12,18,25]}
p = GridSearchCV(estimator=SVC(),param_grid=params,return_train_score=True)



p.fit(xtrain,y_train)

print(p.best_params)
print(p.best_score)

ypred = p.predict(xtest)

cm = confusion_matrix(y_test,ypred)
ac = accuracy_score(y_test,ypred)
cr = classification_report(y_test,ypred)



#from matplotlib.colors import ListedColormap  
#x_set, y_set = xtrain, y_train  
#x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
#nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
#mtp.contourf(x1, x2, model.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
#alpha = 0.75, cmap = ListedColormap(('purple','green' ))
#mtp.xlim(x1.min(), x1.max())  
#mtp.ylim(x2.min(), x2.max())  
#for i, j in enumerate(nm.unique(y_set)):  
    
    #mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        #c = ListedColormap(('purple', 'green'))(i), label = j)  
#mtp.title('SVM (Training set)')  
#mtp.xlabel('KNN')  
#mtp.ylabel('Estimated Salary')  
#mtp.legend()  
#mtp.show()









