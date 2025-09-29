import numpy as np

def model(
    X: np.array,
    W: np.array,
    b: float
):
    return np.dot(W, X) + b

def compute_cost(
    X: np.array,
    y: np.array,
    W: np.array,
    b: float
):
    n, sum = X.shape[0], 0
    for i in range(n):
        error = (np.dot(W, X[i]) + b - y[i]) ** 2 # error between prediction & truth
        sum += error
    return sum / (2.0 * float(n))

def compute_gradient(
    X: np.array,
    y: np.array,
    W: np.array,
    b: float
):
    n, W_gra, b_gra = X.shape[0], np.zeros(X.shape[1]), 0
    for i in range(n):
        for j in range(X.shape[1]):
            W_gra[j] += (np.dot(W, X[i]) + b - y[i]) * X[i][j]
        b_gra += (np.dot(W, X[i]) + b - y[i])

    return W_gra / float(n), b_gra / float(n)

def gradient_descent(
    X: np.array,
    y: np.array, # data set
    a: float,
    W: np.array,
    b: float, # parameters
    iteration: int
):
    assert X.shape[0] == y.shape[0], "invalid train data set"

    for i in range(iteration):
        W_gra, b_gra = compute_gradient(X, y, W, b)
        W = W - a * W_gra
        b = b - a * b_gra

        cost = compute_cost(X, y, W, b)
        if i % 1000 == 0:
            print(f"iteration {i}, w_gra {W_gra}, b_gra {b_gra}, W {W}, b {b}, cost {cost}")
    
    return W, b

# Load our data set
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])   #features
y_train = np.array([460, 232, 178])   #target value
a, W, b, iter = 5.0e-7, np.zeros(X_train.shape[1]), 0.0, 1000

# gradient descent
W, b = gradient_descent(X_train, y_train, a, W, b, iter)
print(f"final W {W}, b {b}")

# predict
for i in range(y_train.shape[0]):
    X = X_train[i]
    print(f"predict X {X}, output y is {model(X, W, b)}, truth {y_train[i]}")
