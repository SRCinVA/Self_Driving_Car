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
import torch.autograd as autograd # for working with tensors (complicated explanation; study this later)


# to build architecture of the NN

# will build the NN as an object (hence, a class)
# number of hidden layers is a key architectural decision
# will use a rectifier function in order to break linearity

class Network(nn.Module):   #Network inherits from the NN module

    def __init__(self, input_size, nb_action):    #standard init function; typical use of <self> (to specify that variables belong to the object created from the class)
                                                    #input_size: an integers of the number of input neurons: three sensors, plus orientation, and minus orientation.
                                                    #three possible action: straight, left, or right) 
        super(Network, self).__init__()               #the super function is from Pytorch and inherits from the NN module. (Question: why the __init__ here?)
        self.input_size = input_size
        self.nb_action = nb_action
        # to build the full connections (fc) between the layers (every neuron is connected to every neuron)
        self.fc1 = nn.Linear()  # to make the connect between between the input layer and the hidden layer
