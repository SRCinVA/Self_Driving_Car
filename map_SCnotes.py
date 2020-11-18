# AI for Self Driving Car

#Libraries to import
import numpy as np #allows use of arrays
import random #helps when pulling samples for experience replay
import os #for loading the model and saving it between sessions (unclear explanation)
import torch #can handle dynamic graphs, unlike some others 
import torch.nn as nn #a component of torch, nn contains the neural network that take in the three sensors' input
                    #will use a softmax in this case to determine action
import torch.nn.functional as F #a shortcut to the functional package of the nn module; elected to use Huber loss function
                            #Huber loss function improves convergence
import torch.optim as optim # an optimizer to perform stochastic gradient descent
import torch.autograd as autograd # for working with tensors (nd arrays)


# to build architecture of the NN

# will build the NN as an object (hence, a class)
# number of hidden layers is a key architectural decision
# will use a rectifier function in order to break linearity

class Network(nn.Module):   #Network inherits from the NN module

    def __init__(self, input_size, nb_action):    #standard init function; typical use of <self> (to specify that variables belong to the object created from the class)
                                                    #input_size: an integers of the number of input neurons: three sensors, plus orientation, and minus orientation.
                                                    #three possible action: straight, left, or right) 
        super(Network, self).__init__()             #the super function is from Pytorch and inherits from the NN module. (Question: why the second __init__ here?)
        self.input_size = input_size
        self.nb_action = nb_action
        # to build the full connections (fc) between the layers (every neuron is connected to every neuron)
        self.fc1 = nn.Linear(input_size, 30)  # to make the connect between between the input layer and the hidden layer
                                    # arguments are: in_features, out_features, bias=True (Question: waht is "bias" here?) 
                                    # key insight: you can tune the architecture by modifying the number of layers and neurons.
        #time to make a 2nd full connection (on the way out):
        self.fc2 = nn.Linear(30, nb_action)

    # this will activate the rectifier function to move the car forward.
    def forward(self, state):       #takes arguments of self and input_size.
        x = F.relu(self.fc1(state)) # x represents the hidden neurons and applies the rectifier (ReLu) function to them. Then we have to pass in "state" to go from input to hidden neurons.
        q_values = self.fc2(x)      # these are q_values from the output neurons, using the "x" from the hidden layer in the previous line.
        return q_values             # returns the q_values for each possible action.