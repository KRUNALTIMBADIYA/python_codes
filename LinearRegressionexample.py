import numpy as np


np.random.seed(42) # for reproducibility
class LinearRegression:
    def __init__(self, lr = 1e-2, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # update weight & bias
            self.weights -= self.lr *dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

class LinearRegressionNormal:
    """Implements a linear regresssion using the normal equation """
    def __init__(self, lr = 1e-2, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) 
        for _ in range(self.n_iters):
            # do a stochastic gradient descent
            for i in range(n_samples):
                random_index = np.random.randint(self.n_iters)
                xi = X[random_index: random_index + 1]
                yi = y[random_index: random_index + 1]
                dw = 2 * xi.T.dot(xi.dot(self.weights) - yi)
                self.weights -= self.lr * dw

    def predict(self, X):
        y_pred = np.dot(X, self.weights)
        return y_pred

if _name_ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(mse)

    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(mse)


    lin_reg = LinearRegressionNormal()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(mse)
