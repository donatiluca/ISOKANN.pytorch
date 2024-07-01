import numpy as np
import torch as pt
from tqdm import tqdm
import random
from modules.other_functions import scale_and_shift
from sklearn.model_selection import train_test_split

# Check if CUDA is available, otherwise use CPU
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

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

    # Early stopping parameters
    # Stop training if this metric no longer improves after a certain number of epochs (patience).
    patience = 10
    best_loss = float('inf')
    best_model = None
    patience_counter = 0

    # Split training and validation data
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Define the optimizer
    optimizer = pt.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    #optimizer = pt.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, nesterov=True, momentum=0.23, dampening=0)

    # Define the loss function
    MSE = pt.nn.MSELoss()
        
    # Define an array where to store the loss
    train_losses = []
    val_losses = []

    # Train the model
    for epoch in range(Nepochs):

        permutation = pt.randperm(X_train.size()[0], device=device)

        for i in range(0, X_train.size()[0], batch_size):
            
            # Clear gradients for next training
            optimizer.zero_grad()
            
            indices = permutation[i:i+batch_size]
            
            batch_x, batch_y = X_train[indices], Y_train[indices]
            
            # Make a new prediction
            new_points  =  net( batch_x )
            
            # measure the MSE
            loss = MSE(batch_y, new_points)

            # computes the gradients of the loss with respect to the model parameters using backpropagation.
            loss.backward()

            # updates the NN parameters
            optimizer.step()

        train_losses.append(loss.item())

        # Validation
        with pt.no_grad():
            val_outputs = net(X_val)
            val_loss    = MSE(val_outputs, Y_val)
            val_losses.append(val_loss.item())

        # Early stopping
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = net.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                #print(f"Early stopping at epoch {epoch+1}")
                break
        
        #print(f'Epoch {epoch+1}/{Nepochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

    return train_losses, val_losses, best_loss


# Random search for hyperparameters
def random_search(X, Y, NN_layers, learning_rates, batch_size=50, search_iterations=20):

    best_hyperparams = None
    best_val_loss = float('inf')

    for _ in tqdm(range(search_iterations)):

        lr    = random.choice(learning_rates)
        nodes = np.asarray(random.choice(NN_layers))

        f_NN = NeuralNetwork( Nodes = nodes ).to(device)

        train_losses, val_losses, val_loss = power_method(X, Y, 
                                                          f_NN, 
                                                          scale_and_shift, 
                                                          Niters = 100, 
                                                          Nepochs = 100,
                                                          tolerance  = 5e-3, 
                                                          batch_size=batch_size,
                                                          lr = lr)

        print("Validation loss:", val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            best_hyperparams = {'nodes': nodes, 'learning_rate': lr}

        del f_NN

    return best_hyperparams, best_val_loss


def power_method(pt_x0, pt_xt, f_NN, scale_and_shift, Niters = 500, Nepochs = 10, tolerance  = 5e-3, batch_size = 50, lr = 1e-3, print_eta=False):

    """
    train_LOSS, val_LOSS, best_loss = power_method(pt_x0, pt_y, f_NN, scale_and_shift, Niters = 500, tolerance  = 5e-3)
    """
    train_LOSS = np.empty(0, dtype = object)
    val_LOSS   = np.empty(0, dtype = object)

    if   print_eta == False:
        loop = range(Niters)
    elif print_eta == True:
        loop = tqdm(range(Niters))
        
    for i in loop:

        old_chi =  f_NN(pt_x0).cpu().detach().numpy()

        pt_chi  =  f_NN( pt_xt )
        pt_y    =  pt.mean(pt_chi, axis=1)
        y       =  scale_and_shift(pt_y.detach().cpu().detach().numpy())
        pt_y    =  pt.tensor(y, dtype=pt.float32, device = device)
        
        train_loss, val_loss, best_loss = trainNN(net = f_NN, lr = lr, wd = 1e-5, Nepochs = Nepochs, batch_size=batch_size, X=pt_x0, Y=pt_y)
        train_LOSS           = np.append(train_LOSS, train_loss[-1])
        val_LOSS             = np.append(val_LOSS, val_loss[-1])

        new_chi   = f_NN(pt_x0).cpu().detach().numpy()
        #print(np.linalg.norm(new_chi - old_chi) )
        if np.linalg.norm(new_chi - old_chi) < tolerance:
            
            break

    return train_LOSS, val_LOSS, best_loss



