# AI for Self Driving Car

#Libraries to import
import numpy as np #allows use of arrays
import random #helps when pulling samples for experience replay
import os #for loading the model and saving it between sessions (unclear explanation)
import torch #can handle dynamic graphs, unlike some others 
import torch.nn #a component of torch, nn contains the neural network that take in the three sensors' input
                #will use a softmax in this case to determine action
import torch.nn.functional as F #a shortcut to the functional package of the nn module; elected to use Huber loss function
                            #Huber loss function improves convergence
import torch.optim as optim # an optimizer to perform stochastic gradient descent
import torch.autograd as autograd # for working with tensors (complicated explanation; study this later)