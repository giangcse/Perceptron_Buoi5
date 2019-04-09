import pandas as pd 
data = pd.read_csv('data_per.csv')

X = data[['X1', 'X2', 'X3', 'X4', 'X5']]
y = data['Y']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=1)

from sklearn.linear_model import Perceptron
net = Perceptron()
net.fit(X_train, y_train)
print (net)
net.coef_ #Các trọng số thuộc tính (w1 -> w5)
net.intercept_ # w0
net.n_iter_ # số lần lặp
#net.score(X, y)*100

y_pred = net.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, y_pred)*100
