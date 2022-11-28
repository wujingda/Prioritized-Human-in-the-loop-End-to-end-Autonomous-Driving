import sys
import pygame
def signal_handler(sig, frame):
    print('Procedure terminated!')
    pygame.display.quit()
    pygame.quit()
    sys.exit(0)

from scipy.interpolate import interp1d

## This function provides a prospective lateral-coordinate generator w.r.t possible longitudinal coordinates
## for the ego vehicle in Scenario 0, which can be taken as a demonstration 
def get_path():
    waypoint_x_mark = np.array([-13.0001,-13.0001,-13.0004,-13.0006,-13.0002,-13.0013,-12.9994,-12.9972,-12.9952,-12.9937,-12.9842,-12.9727,-12.9635,-12.958,-12.9382,-12.9204,-12.9032,-12.8978,-12.887,-12.8741,-12.8609,-12.8481,-12.8461,-12.8461])
    waypoint_y_mark = np.array([-200,-156.33,-152.294,-148.325,-144.192,-142.402,-137.921,-133.517,-129.183,-127.453,-123.262,-118.943,-115.795,-113.691,-108.998,-104.562,-100.227,-98.5793,-94.3961,-90.0766,-85.7045,-81.477,-80.5685,0])
    pathgenerator = interp1d(waypoint_y_mark, waypoint_x_mark,kind='linear')
    return pathgenerator

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1,6,6)
        self.conv2 = nn.Conv2d(6,16,6)
        self.fc = nn.Linear(16*7*16, 128)
    
    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.max_pool2d( self.conv1(x),2)
        x = F.max_pool2d( self.conv2(x),2)
        x = x.view(x.size(0),16*16*7)
        
        x = self.fc(x)
        x = self.relu(x)
        
        return x
        
        
class RND(nn.Module):
    def __init__(self, use_cuda = True):
        super(RND, self).__init__()
        
        self.use_cuda = use_cuda
        self.fix = NET()
        self.estimator = NET()
        
        self.fix.apply(weights_init_normal)
        
        self.estimator.apply(weights_init_normal)
        
        self.criterion = nn.MSELoss()
        self.optim = optim.Adam(self.estimator.parameters(),0.0001)
        
        if self.use_cuda:
            self.fix.cuda()
            self.estimator.cuda()
        
    def forward(self, state):
        
        state = torch.tensor(state, dtype=torch.float).reshape(1,45,80)
        if self.use_cuda:
            state = state.cuda()
        
        target = self.fix.forward(state)
        estimate = self.estimator.forward(state)
        
        loss = self.criterion(estimate, target)
        self.optim.zero_grad()
        
        loss.backward()
        self.optim.step()
        
        error = loss.item()
        mu = torch.mean(target)
        std = torch.std(target)
        
        return error, mu.detach().cpu().numpy(), std.detach().cpu().numpy()
    
    def get_reward_i(self, state):
        
        error, mu, std = self.forward(state)
        
        alpha = 1+(error-mu)/std
        
        x = min( max(alpha, 1), 2)
        
        return x
        
