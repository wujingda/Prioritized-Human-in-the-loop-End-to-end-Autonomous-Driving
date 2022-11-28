import torch
import torch.nn as nn
import torch.nn.functional as F

### This actor network is established on CNN structure for the image-style state with the shape of 80*45.
class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden=32, init_w=3e-1):
        super(Actor,self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 6)
        self.conv2 = nn.Conv2d(6, 16, 6)
        self.fc1 = nn.Linear(16*16*7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, nb_actions)
        self.sig = nn.Tanh()
        self.init_weights(init_w)
        
    def init_weights(self,init_w):

        torch.nn.init.kaiming_uniform_(self.fc1.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')    
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.fc4.weight.data.uniform_(-init_w,init_w)
        
    def forward(self,inp):
        
        x = inp.unsqueeze(0) if len(inp.shape)==3 else inp
        x = F.max_pool2d( self.conv1(x), 2)
        x = F.max_pool2d( self.conv2(x), 2)
        x = x.reshape(x.size(0),-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sig(self.fc4(x))

        return x

### This critic network is established on MLP structure.
class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden=256, init_w=3e-1):
        super(Critic,self).__init__()

        self.conv1 = nn.Conv2d(2, 6, 6)
        self.conv2 = nn.Conv2d(6, 16, 6)
        self.fc1 = nn.Linear(16*16*7 + nb_actions, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, nb_actions)
        
        self.fc11 = nn.Linear(16*16*7 + nb_actions, 256)
        self.fc21 = nn.Linear(256, 128)
        self.fc31 = nn.Linear(128, 64)
        self.fc41 = nn.Linear(64, nb_actions)

        self.init_weights(init_w)
        
    def init_weights(self,init_w):

        torch.nn.init.kaiming_uniform_(self.fc1.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.fc4.weight.data.uniform_(-init_w,init_w)
        
        torch.nn.init.kaiming_uniform_(self.fc11.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc21.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc31.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.fc41.weight.data.uniform_(-init_w,init_w)
        
    def forward(self,inp):
        
        x, a = inp
        x = x.unsqueeze(0) if len(x.shape)==3 else x
        x = F.max_pool2d( self.conv1(x), 2)
        x = F.max_pool2d( self.conv2(x), 2)
        x = x.reshape(x.size(0),-1)
        
        q1 = F.relu(self.fc1(torch.cat([x,a],1)))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = self.fc4(q1)
        
        q2 = F.relu(self.fc11(torch.cat([x,a],1)))
        q2 = F.relu(self.fc21(q2))
        q2 = F.relu(self.fc31(q2))
        q2 = self.fc41(q2)
        
        return q1, q2


    
    
    
    
