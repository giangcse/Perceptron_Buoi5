import pandas as pd 
data = pd.read_csv('data_per.csv')

X = data[['X1', 'X2', 'X3', 'X4', 'X5']]
y = data['Y']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train[1:5]
y_train[1:5]
X_test[1:5]
y_test[1:5]

from sklearn.linear_model import Perceptron
net = Perceptron()
net.fit(X_train, y_train)
print (net)
net.coef_
net.intercept_
net.n_iter_
