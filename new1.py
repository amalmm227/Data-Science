import pandas as pd
import pickle

f1=open(file="naivebayse.pkl",mode="br")
m1 = pickle.load(f1)
f1.close()

f2=open(file="stndrd.pkl",mode="br")
m2 = pickle.load(f2)
f2.close()

def prediction(a,b):
    data = {'age':a,'estimated_salary':b}
    
    df = pd.DataFrame(data,index=[0])
    df1 =m2.transform(df)
    pred = m1.predict(df1)
    
    if int(pred)==1:
        return 'purschases'
    else:
        return 'not purchased'
    
age = int(input('Enter your age:'))
 
estimated_salary = int(input('Enter your salary:'))
                   
                   
prediction(age,estimated_salary)                   
