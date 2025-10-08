import numpy as np
import mnist_loader

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class NetWork():
    def __init__(self, sizes: list):
        self.num_layers = len(sizes)
        self.sizes = sizes
        ''' The input layer does not have a bias term.
        y represents the output dimension (m), while x represents the input dimension (k).'''
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a: np.array):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.matmul(w, a) + b)
        return a
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        import pdb
        pdb.set_trace()
        results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            # if np.dot(w, activation).shape[1] != b.shape[1]:
            # import pdb
            # pdb.set_trace()
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nbala_w = [np.zeros_like(weight) for weight in self.weights]
        nbala_b = [np.zeros_like(bias) for bias in self.biases]
        for x, y in mini_batch:
            # Compute the gradient of the weights and biases with respect to a single data point.
            delta_nbala_b, delta_nbala_w = self.backprop(x, y)
            nbala_w = [nw+dnw for (nw, dnw) in zip(nbala_w, delta_nbala_w)]
            nbala_b = [nb+dnb for (nb, dnb) in zip(nbala_b, delta_nbala_b)]
        
        self.weights = [w - (eta/len(mini_batch))*nw for (w, nw) in zip(self.weights, nbala_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for(b, nb) in zip (self.biases, nbala_b)]
    
    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent. The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs. The other non-optional parameters are
        self-explanatory. If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out. This is useful for
        tracking progress, but slows things down substantially."""

        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                    for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {j} complete.")

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = NetWork([784, 16, 10])
net.stochastic_gradient_descent(training_data=list(training_data), epochs=30, mini_batch_size=10, eta=3.0, test_data=list(test_data))