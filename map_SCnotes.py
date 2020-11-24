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

# create a class for experience replay aspect
# this will help us understand 100 states in the past (!!), creating a long-term memory. It helps Deep Q learning.
# then it takes random samples to make the next move

class ReplayMemory(object):  # Question: what exactly does 'object' refer to here?

    def __init__(self, capacity): #capacity is 100, based on our choice to keep 100 last transitions.
        self.capacity = capacity
        self.memory = []          # this initializes the memory, and from there, the list will be added to

    # the push function will (1.) append a new transition to the memory and (2.) ensure that the memory has 100 transitions 
    def push(self, event):
        self.memory.append(event) # Question: when will we define the event here in the code ...?
        if len(self.memory) > self.capacity:
            del self.memory[0]    # delete the oldest memory (which naturally lives at index [0])
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))  # we take random samples from the memory that have a fixed size of batch_size.
                                                                # zip * reshapes your list of tuples, to bucket states, actions, and rewards together.
                                                                # these batches then get wrapped into a PyTorch variable, which contains a tensor and a gradient. 
                                                                # This format enables PyTorch variables to take in the information.
        return map(lambda x: Variable(torch.cat(x,0)), samples) # Question: very difficult explanation here

class Dqn(): # Implementing Deep Q Learning model
    def __init__(self, input_size, nb_action, gamma):    #the input_size, the number of possible actions, and gamma)
        self.gamma = gamma
        self.reward_window = []     # a sliding window of the evolving mean of the last 100 rewards
                                    # we initalize reward_window as an empty list
        self.model = Network(input_size, nb_action) # this creates one instance of an NN for the Dqn class (a class within a class, it seems?)
        self.memory = ReplayMemory(100000) # we'll take 100,000 transitions into memory and sample them, enabling the model to learn.
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)      # Based on torch.optim for stochastic gradient descent. 'Adam' is one such optimizer and work well for our purposes.
                                                                                # The arguments are all the paramaters to customize the optimizer.
                                                                                # We'll pass in self.model, with '.parameters' to accept those parameters. 
                                                                                # In sum, this line of code connects the Adam optimizer to the neural network.
                                                                                # lr = learning rate
        # now we need to build the last three variables composing the transition events:
        self.last_state = torch.Tensor(input_size).unsqueeze(0)     # min 13-14, highly involved explanation of how batching becomes a "fake" vector for use in PyTorch.
                                                                    # We need to position the fake dimension as the first index of the last state, using '.unsqueeze'.
                                                                    # the first dimension [0] is the fake one, and the Tensor from PyTorch will contain the other five.
                                                                    
        self.last_action = 0  # We can intialize this just to zero; no need to create a vector here.
        self.last_reward = 0  # We can initialize this to zero as well.

    def select_action(self, state): # 'state' is in fact the output of the neural network (the Q-values of the three possible actions) 
        probs = F.softmax(self.model(Variable(state, volatile = True))*7)   # the Softmax fucntion goes for the optimum outcome while allowing exploration (by alloting a high probability to the highest Q-value).
                            # Notice that the Softmax function lives in the 'F' module in PyTorch
                            # For the Softmax function, we input the entities for which we want to generate the probablity distribution (the Q values).
                            # Now, all we have to do to make that happen is input the model.
                            # 'state' is a torch tensor, which we'll convert into a variable (Question: why?)
                            # Incomprehensible explanation for 'volatile = True' at around 8:16 in Step 10 video.
                            # the temperature parameter helps the NN decide which action to take.
                            # it's positive number; the higher it is, the likelier it is the car will following the winning Q-value.
                            # 