import pickle
import numpy as np

import torch
import torch.nn as nn

from cpprb import ReplayBuffer
from algo.SLnetwork import Actor


## Hyperparameters
MEMORY_CAPACITY = 38400
BATCH_SIZE = 128
LR = 0.0002

class SL:
        
    def __init__(self, action_dim, state_dim, LR = LR, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        self.device = device
        
        # Hyperparameters configuration
        self.state_dim = state_dim[0] * state_dim[1]
        self.state_dim_width = state_dim[0]
        self.state_dim_height = state_dim[1]
        self.action_dim = action_dim
        self.batch_size = BATCH_SIZE

        # Priority Experience Replay Buffer
        self.pointer = 0
        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY,
                                                  {"obs": {"shape": (45,80)},
                                                   "act": {"shape":action_dim}}
                                                  )
        
        self.actor = Actor(self.state_dim,self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LR)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, 0.9996)
        self.previous_epoch = 0
        

    def learn(self, batch_size = BATCH_SIZE, epoch=0):

        ## batched state, batched action, batched action from expert, batched intervention signal, batched reward, batched next state
        data = self.replay_buffer.sample(batch_size)
        bs, ba = data['obs'], data['act']
        bs = torch.FloatTensor(bs).permute(0,3,1,2).to(self.device)
        ba = torch.FloatTensor(ba).to(self.device)

        ## calculate the predicted values of the actor
        pred_a = self.actor.forward(bs)
        
        # calculate the supervised learning loss
        loss = ((pred_a - ba)**2).mean()

        # update the actor
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        if epoch != self.previous_epoch:
            self.actor_scheduler.step() 
            self.previous_epoch = epoch
            
        return loss


    def choose_action(self,state):
        state = torch.tensor(state, dtype=torch.float).reshape(self.state_dim_height, self.state_dim_width).to(self.device)
        state = state.unsqueeze(0)
        
        action = self.actor.forward(state).detach().cpu().numpy()
        action = np.clip(action,-1,1)

        return action.squeeze()
    

    def store_transition(self,  s, a):
        self.replay_buffer.add(obs=s,
                               act=a)
    

    def save_transition(self, output, timeend=0):
        self.replay_buffer.save_transitions(file='{}/{}'.format(output, timeend))

    def load_transition(self, output):
        if output is None: return
        self.replay_buffer.load_transitions('{}.npz'.format(output))
            
            
    def load_model(self, output):
        if output is None: return
        self.actor.load_state_dict(torch.load(output,map_location='cuda:0'))


    def save_model(self, output, seed=0):
        torch.save(self.actor.state_dict(), '{}/SL{}.pkl'.format(output, seed))

