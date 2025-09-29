import numpy as np

def model(
    x: float,
    w: float,
    b: float
):
    return w * x + b

def compute_cost(
    x: np.array,
    y: np.array,
    w: float,
    b: float
):
    m, sum = x.shape[0], 0
    for i in range(m):
        error = (w * x[i] + b - y[i]) ** 2 # error between prediction & truth
        sum += error
    return sum / (2.0 * float(m))

def compute_gradient(
    x: np.array,
    y: np.array,
    w: float,
    b: float
):
    m, w_gra, b_gra = x.shape[0], 0, 0
    for i in range(m):
        w_gra += (w * x[i] + b - y[i]) * x[i]
        b_gra += (w * x[i] + b - y[i])

    return w_gra / float(m), b_gra / float(m)

def gradient_descent(
    x: np.array,
    y: np.array, # data set
    a: float,
    w: float,
    b: float, # parameters
    iteration: int
):
    assert x.shape[0] == y.shape[0], "invalid train data set"

    for i in range(iteration):
        w_gra, b_gra = compute_gradient(x, y, w, b)
        w = w - a * w_gra
        b = b - a * b_gra

        cost = compute_cost(x, y, w, b)
        if i % 1000 == 0:
            print(f"iteration {i}, w_gra {w_gra}, b_gra {b_gra}, w {w}, b {b}, cost {cost}")
    
    return w, b

# Load our data set
x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value
a, w, b, iter = 0.5, 0.0, 0.0, 2000

# gradient descent
w, b = gradient_descent(x_train, y_train, a, w, b, iter)
print(f"final w {w}, b {b}")

# predict
x = 2.0
print(f"predict x {x}, output y is {model(x, w, b)}")
