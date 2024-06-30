import torch as pt
import numpy as np
import scipy
from scipy import stats

# Check if CUDA is available, otherwise use CPU
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"Torch is on: {device}")


class NeuralNetwork(pt.nn.Module):
    def __init__(self, Nodes, enforce_positive=0):
        super(NeuralNetwork, self).__init__()

        # self parameters
        self.input_size        = Nodes[0]
        self.output_size       = Nodes[-1]
        self.NhiddenLayers     = len(Nodes) - 2
        self.Nodes             = Nodes

        self.enforce_positive  = enforce_positive

        # build NN architecture
        self.hidden_layers = pt.nn.ModuleList()

        # add layers
        self.hidden_layers.extend([pt.nn.Linear(self.input_size,    self.Nodes[1])])
        self.hidden_layers.extend([pt.nn.Linear(self.Nodes[1+l], self.Nodes[1+l+1]) for l in range(self.NhiddenLayers)])

        # the output of the last layer must be equal to 1
        #if self.Nodes[-1] > 1:
        #    self.hidden_layers.extend([pt.nn.Linear(self.Nodes[-1], 1)])

        # define activation function
        self.activation1  = pt.nn.Sigmoid()  # #
        self.activation2  = pt.nn.ReLU()
        self.activation3  = pt.nn.Softplus(10)

    def forward(self, X):

        # Pass input through each hidden layer but the last one
        for layer in self.hidden_layers[:-1]:
            X = self.activation1(layer(X))

        # Apply the last layer (but not the activation function)
        X = self.hidden_layers[-1](X)

        if self.enforce_positive == 1:
            X= self.activation3(X)  #.unsqueeze(1)

        return X.squeeze()

def trainNN(net, lr, wd, Nepochs, batch_size, X, Y):
    # Define the optimizer
    optimizer = pt.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    # Define the loss function
    MSE = pt.nn.MSELoss()

    # Define an array where to store the loss
    loss_arr = np.zeros(Nepochs)

    # Train the model
    for epoch in range(Nepochs):

        permutation = pt.randperm(X.size()[0], device=device)

        for i in range(0, X.size()[0], batch_size):

            # Clear gradients for next training
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X[indices], Y[indices]

            # Make a new prediction
            new_points  =  net( batch_x )

            # measure the MSE
            loss = MSE(batch_y, new_points)

            # computes the gradients of the loss with respect to the model parameters using backpropagation.
            loss.backward()

            # updates the NN parameters
            optimizer.step()

        loss_arr[epoch] = loss.item()

    return loss_arr



