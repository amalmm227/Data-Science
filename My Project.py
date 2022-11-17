#!/usr/bin/env python
# coding: utf-8

# In[432]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[433]:


df1=pd.read_csv('car data.csv')
df1.head(11)


# In[434]:


df1.shape


# In[435]:


df1.info()


# In[436]:


df1.isnull().sum()


# In[437]:


df1['Car_Name'].unique()


# In[438]:


df1['Current_Yr']=2022
df1.head()


# In[439]:


df1['Age']=df1['Current_Yr']-df1['Year']
df1.head()


# In[440]:


df2=df1.drop(['Year','Current_Yr','Present_Price'],axis='columns')
df2.head()


# In[441]:


from sklearn.preprocessing import LabelEncoder


# In[442]:


le_Fuel_Type=LabelEncoder()
le_Seller_Type=LabelEncoder()
le_Transmission=LabelEncoder()


# In[443]:


df2['Fuel_Type_N']=le_Fuel_Type.fit_transform(df2['Fuel_Type'])
df2['Seller_Type_N']=le_Seller_Type.fit_transform(df2['Seller_Type'])
df2['Transmission_N']=le_Transmission.fit_transform(df2['Transmission'])
df2.head()


# In[444]:


df3=df2.drop(['Fuel_Type','Seller_Type','Transmission',],axis=1)
df3.head()


# In[445]:


df3['Selling_Price'].min()


# In[446]:


df3['Selling_Price']=df3['Selling_Price']*100000
df3.head()


# In[447]:


df3['Selling_Price']=df3['Selling_Price'].astype('int32')
df3.head()


# In[448]:


plt.bar(df3['Car_Name'],df3['Selling_Price'])
plt.show()


# In[449]:


df3['Selling_Price'].unique()


# In[450]:


df3.isna().sum()


# In[451]:


df3['Selling_Price'].max()


# In[452]:


df3['Selling_Price'].min()


# In[453]:


len(df3['Selling_Price'])


# In[454]:


df3.head()


# In[455]:


df3.Car_Name=df3.Car_Name.apply(lambda x:x.strip())


# In[456]:


Counts=df3.Car_Name.value_counts(ascending=False)
Counts


# In[457]:


dummies=pd.get_dummies(df3['Car_Name'])
dummies.head()


# In[458]:


dummies1=dummies.drop(['xcent'],axis=1)
dummies1.head()


# In[459]:


df4=pd.concat([df3,dummies1],axis='columns')
df4.head()


# In[460]:


df5=df4.drop(['Car_Name'],axis=1)
df5.head()


# In[461]:


df5.shape


# In[462]:


x=df5.drop(['Selling_Price'],axis=1)
x.head()


# In[463]:


y=df5['Selling_Price']
y.head()


# In[464]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)


# In[465]:


from sklearn import linear_model
model=linear_model.LinearRegression()
model.fit(x_test,y_test)


# In[466]:


model.score(x_test,y_test)


# In[467]:


x.columns


# In[468]:


df4['Car_Name'].unique()


# In[469]:


Car_Names='800'
np.where(x.columns==Car_Names)[0][0]


# In[470]:


x.columns


# In[473]:


def predict_price(Car_Name,Kms_Driven,Owner,Age,Fuel_Type_N,Seller_Type_N,Transmission_N):    
    loc_index = np.where(x.columns==Car_Names)[0][0]

    x = np.zeros(len(x.columns)) 
    x[0] = Car_Name
    x[1] = Kms_Driven
    x[2] = Owner
    x[3] = Age
    x[4] = Fuel_Type_N
    x[5] = Seller_Type_N
    x[6] = Transmission_N
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]


# In[472]:


x.head()


# In[474]:


predict_price('800',27000,0,8,2,0,1)

