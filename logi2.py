import  
2.	from matplotlib.colors import ListedColormap  
3.	x_set, y_set = x_train, y_train  
4.	x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
5.	nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
6.	mtp.contourf(x1, x2, classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
7.	alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
8.	mtp.xlim(x1.min(), x1.max())  
9.	mtp.ylim(x2.min(), x2.max())  
10.	for i, j in enumerate(nm.unique(y_set)):  
11.	    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
12.	        c = ListedColormap(('purple', 'green'))(i), label = j)  
13.	mtp.title('Logistic Regression (Training set)')  
14.	mtp.xlabel('Age')  
15.	mtp.ylabel('Estimated Salary')  
16.	mtp.legend()  
17.	mtp.show()  
18.	

On Wed, Nov 23, 2022 at 11:44 AM Irfana Khadheeja <irfanakhadeeja.futura@gmail.com> wrote:



