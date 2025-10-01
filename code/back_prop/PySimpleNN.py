import numpy as np
import matplotlib.pyplot as plt
import time
import os

def stochastic_update_Adam(w, grad_w, mt, vt,lrate, iteration):
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;

    mt = mt*beta1 + (1.0-beta1)*grad_w;
    vt = vt*beta2 + (1.0-beta2)*grad_w**2;

    mt_hat = mt/(1.0-beta1**iteration);
    vt_hat = vt/(1.0-beta2**iteration);

    scal = 1.0/(np.sqrt(vt_hat) + epsilon);

    w = w - lrate*mt_hat*scal;
    
    return w, mt, vt

def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return np.random.normal(size = [in_dim, out_dim], scale=xavier_stddev)

def initialize_NN(layers):
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = np.zeros([1,layers[l+1]]) 
        weights.append(W)
        biases.append(b)
    return weights, biases

class NN:
    # Initialize the class
    def __init__(self, X, Y, layers):
        
        # data
        self.X = X # N x P
        self.Y = Y # N x Q
        
        # layers
        self.layers = layers # P ----> Q
        
        # initialize NN
        self.weights, self.biases = initialize_NN(layers)
    
    def loss_function(self):
        num_layers = len(self.layers)
        weights = self.weights
        biases = self.biases
        
        H = [None]*(num_layers-1)
        A = [None]*(num_layers-1)
        
        H[0] = self.X
        A[0] = self.X
        
        for l in range(0,num_layers-2):
            H[l+1] = np.dot(A[l],weights[l]) + biases[l]
            A[l+1] = np.tanh(H[l+1])
        
        Y_pred = np.dot(A[num_layers-2],weights[num_layers-2]) + \
                 biases[num_layers-2]
        
        loss = 0.5*np.sum((Y_pred-self.Y)**2)
        
        ### backpropagation
        
        G = [None]*(num_layers-1)
        loss_weights = [None]*(num_layers-1)
        loss_biases = [None]*(num_layers-1)
        
        G[num_layers-2] = Y_pred - self.Y
        
        for l in range(num_layers-2,0,-1):
            
            loss_weights[l] = np.dot(H[l].T, G[l])
            loss_biases[l] = np.sum(G[l],axis=0)
            
            G[l-1] = (1 - A[l]**2)*np.dot(G[l],weights[l].T)
        
        loss_weights[0] = np.dot(H[0].T, G[0])
        loss_biases[0] = np.sum(G[0],axis=0)
        
        return loss, loss_weights, loss_biases, Y_pred
    
    def train(self, max_iter, learning_rate):
        
        num_layers = len(self.layers)
        
        # Adam Optimizer Parameters        
        mt_weights, mt_biases = [None]*(num_layers-1), [None]*(num_layers-1)
        vt_weights, vt_biases = [None]*(num_layers-1), [None]*(num_layers-1)
        
        for l in range(0,num_layers-1):
            mt_weights[l], mt_biases[l] = [np.zeros(self.weights[l].shape),
                                          np.zeros(self.biases[l].shape)]
            vt_weights[l], vt_biases[l] = [np.zeros(self.weights[l].shape),
                                           np.zeros(self.biases[l].shape)]
        
        start_time = time.time()
        for it in range(1,max_iter+1):
            # Compute loss and gradients 
            loss, loss_weights, loss_biases, Y_pred = self.loss_function()
            
            # Update parameters
            for l in range(0,num_layers-1):
                [self.weights[l],
                 mt_weights[l],
                 vt_weights[l]] = stochastic_update_Adam(self.weights[l],
                                                         loss_weights[l],
                                                         mt_weights[l],
                                                         vt_weights[l],
                                                         learning_rate, it)
                
                [self.biases[l],
                 mt_biases[l],
                 vt_biases[l]] = stochastic_update_Adam(self.biases[l],
                                                        loss_biases[l],
                                                        mt_biases[l],
                                                        vt_biases[l],
                                                        learning_rate, it)
                
            if it % 5000 == 0:
                elapsed = time.time() - start_time
                print('Iteration: %d, loss: %.5f, Time: %.5f, learning rate: %.1e' % (it, loss, elapsed, learning_rate))
                start_time = time.time()
    
    def test(self,X_star):
        num_layers = len(self.layers)
        weights = self.weights
        biases = self.biases
        
        H = X_star
        A = X_star
        
        for l in range(0,num_layers-2):
            H = np.dot(A,weights[l]) + biases[l]
            A = np.tanh(H)
        
        Y_pred = np.dot(A,weights[num_layers-2]) + biases[num_layers-2]
        
        return Y_pred
    
    def save(self, filename):
        # 保存每层权重和偏置为单独的文件项
        np.savez(
            filename,
            **{f'weight_{i}': w for i, w in enumerate(self.weights)},
            **{f'bias_{i}': b for i, b in enumerate(self.biases)}
        )

    def load(self, filename):
        data = np.load(filename)
        num_layers = len(self.weights)
        self.weights = [data[f'weight_{i}'] for i in range(num_layers)]
        self.biases = [data[f'bias_{i}'] for i in range(num_layers)]
    
import matplotlib.pyplot as plt

def draw_nn(layers):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    y_gap = 1.2
    x_gap = 1.5
    max_neurons = max(layers)
    for i, n_neurons in enumerate(layers):
        x = i * x_gap
        y_start = (max_neurons - n_neurons) / 2 * y_gap
        for j in range(n_neurons):
            y = y_start + j * y_gap
            circle = plt.Circle((x, y), 0.3, color='skyblue', ec='black')
            ax.add_patch(circle)
            if i > 0:
                prev_n = layers[i-1]
                prev_y_start = (max_neurons - prev_n) / 2 * y_gap
                for k in range(prev_n):
                    prev_y = prev_y_start + k * y_gap
                    ax.plot([x-x_gap+0.3, x-0.3], [prev_y, y], color='gray', lw=0.5)
    plt.title("mlp")
    plt.savefig('mlp.png')
    
if __name__ == "__main__": 
    
    training_data = np.loadtxt("training_data.csv")
        
    X_train = training_data[:,0:1]
    Y_train = training_data[:,1:2]
    
    model = NN(X_train, Y_train, layers=[1]+10*[10]+[1])
    draw_nn([1]+10*[10]+[1])
    
    model_file = './mlp.npz'
    if not os.path.exists(model_file):
        print(f"start model training")
        model.train(max_iter=10000, learning_rate=1e-3)
        model.train(max_iter=10000, learning_rate=1e-4)
        model.train(max_iter=10000, learning_rate=1e-5)
        model.train(max_iter=10000, learning_rate=1e-6)
        model.save(model_file)
    else:
        print(f"loda model from {model_file}")
        model.load(model_file)

    test_data = np.loadtxt("test_data.csv")
    
    X_test = test_data[0:-1:100,0:1]
    Y_test = test_data[0:-1:100,1:2]

    plt.figure()
    plt.plot(X_train, Y_train, 'kx', label='Train data')
    plt.plot(X_test, Y_test, 'b^', label='Test data (truth)')
        
    Y_pred = model.test(X_test)
    
    cost = np.linalg.norm(Y_pred - Y_test)/np.linalg.norm(Y_test)
    print('Cost %e' %(cost))
    
    plt.plot(X_test, Y_pred, 'r.', label='Prediction')
    residual = Y_pred - Y_test
    plt.plot(X_test, residual, 'go', label='Residual (truth - prediction)')
    plt.legend()
    plt.savefig('backprop.png')