#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('diabetes.csv')
df


# In[3]:


df.shape


# In[4]:


df.info()


# In[6]:


df.isna().sum()


# In[8]:


df.describe()


# In[9]:


df.head()


# In[26]:


df['DiabetesPedigreeFunction'].unique()


# In[41]:


x = df.iloc[:,0:8].values
x


# In[42]:


y = df.iloc[:,8].values
y


# In[48]:


from sklearn.preprocessing import StandardScaler


# In[49]:


st = StandardScaler()


# In[52]:


X = st.fit_transform(x)
X


# In[54]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=30)


# In[55]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)


# In[56]:


model.fit(x_train,y_train)


# In[57]:


ypred = model.predict(x_test)
ypred


# In[58]:


print(y_test)


# In[65]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[68]:


cm = confusion_matrix(y_test,ypred)
cr=classification_report(y_test,ypred)
ac=accuracy_score(y_test,ypred)




#Visualizing the training set result  
from matplotlib.colors import ListedColormap  
x_set, y_set = x_train, y_train  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, model.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
    c = ListedColormap(('purple', 'green'))(i), label = j)  
plt.title('Logistic Regression (Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()  






