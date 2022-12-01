
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cpprb import PrioritizedReplayBuffer

from algo.network_model import Actor,Critic
from algo.util import hard_update, soft_update


MEMORY_CAPACITY = 38400
BATCH_SIZE = 128
GAMMA = 0.95
LR_C = 0.0005
LR_A = 0.0002
LR_I = 0.01
TAU = 0.001
POLICY_NOSIE = 0.2
POLICY_FREQ = 1
NOISE_CLIP = 0.5

class DRL:
        
    def __init__(self, action_dim, state_dim, device='cuda', LR_C = LR_C, LR_A = LR_A):

        self.device = device
        
        self.state_dim = state_dim[0] * state_dim[1] * 2
        self.state_dim_width = state_dim[0]
        self.state_dim_height = state_dim[1]
        self.action_dim = action_dim
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.policy_noise = POLICY_NOSIE
        self.noise_clip = NOISE_CLIP
        self.policy_freq = POLICY_FREQ
        self.itera = 0

        self.pointer = 0
        self.replay_buffer = PrioritizedReplayBuffer(MEMORY_CAPACITY,
                                                  {"obs": {"shape": (45,80,2)},
                                                   "act": {"shape":action_dim},
                                                   "acte": {"shape":action_dim},
                                                   "intervene": {},
                                                   "rew": {},
                                                   "next_obs": {"shape": (45,80,2)},
                                                   "done": {}},
                                                  next_of=("obs"))
        
        self.actor = Actor(self.state_dim,self.action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim,self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LR_A)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, 0.996)
        self.previous_epoch = 0
        
        self.critic = Critic(self.state_dim,self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim,self.action_dim).to(self.device)
        self.critic_optimizers = torch.optim.Adam(self.critic.parameters(),LR_C)
        
        hard_update(self.actor_target,self.actor)
        hard_update(self.critic_target,self.critic)
        
            
    def learn(self, batch_size = BATCH_SIZE, epoch=0):

        ## batched state, batched action, batched action from expert, batched intervention signal, batched reward, batched next state
        data = self.replay_buffer.sample(batch_size)
        idxs = data['indexes']
        states, actions, actions_exp = data['obs'], data['act'], data['acte']
        interv, rewards = data['intervene'], data['rew']
        next_states, dones = data['next_obs'], data['done']
        
        states = torch.FloatTensor(states).permute(0,3,1,2).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        actions_exp = torch.FloatTensor(actions_exp).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).permute(0,3,1,2).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # initialize the loss variables
        loss_c, loss_a = 0, 0

        ## calculate the predicted values of the critic
        with torch.no_grad():
            noise1 = (torch.randn_like(actions) * self.policy_noise).clamp(0, 1)
            next_actions = (self.actor_target(next_states).detach() + noise1).clamp(0, 1)
            target_q1, target_q2 = self.critic_target([next_states, next_actions])
            target_q = torch.min(target_q1,target_q2)
            y_expected = rewards + (1-dones)*self.gamma * target_q    
        y_predicted1, y_predicted2 = self.critic.forward([states, actions]) 
        
        ## calculate td error
        td_errors = abs(y_expected - y_predicted1.detach())
        
        ## update the critic
        loss_critic = F.mse_loss(y_predicted1,y_expected) + F.mse_loss(y_predicted2,y_expected)
        self.critic_optimizers.zero_grad()
        loss_critic.backward()
        self.critic_optimizers.step()

        ## update the actor
        if self.itera % self.policy_freq == 0:
            
            index_imi, _ = np.where(interv==1)
            states_imi = states[index_imi]
            actions_imi = actions[index_imi]
            pred_actions = self.actor.forward(states)
            
            if len(index_imi) > 0:
                imitation_loss = 3 * ((self.actor.forward(states_imi) - actions_imi)**2).sum()
            else:
                imitation_loss = 0.
            
            loss_actor = -self.critic([states, pred_actions])[0] + imitation_loss
            loss_actor = loss_actor.mean()
            
            
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()

            if epoch != self.previous_epoch:
                self.actor_scheduler.step() 
            self.previous_epoch = epoch
            
            soft_update(self.actor_target,self.actor,self.tau)
            soft_update(self.critic_target,self.critic,self.tau)

            loss_a = loss_actor.item()

        loss_c = loss_critic.item()
        
        self.itera += 1
        
        priorities = td_errors.cpu().numpy()

        self.replay_buffer.update_priorities(idxs, priorities)

        return loss_c, loss_a
    
                
    def choose_action(self,state):

        state = torch.FloatTensor(state).float().unsqueeze(0).permute(0,3,1,2).to(self.device)
        
        action = self.actor.forward(state).detach()
        action = action.squeeze(0).cpu().numpy()
        action = np.clip(action,-1, 1)

        return action
    

    def store_transition(self,  s, a, ae, i, r, s_, d=0):
        self.replay_buffer.add(obs=s,
                               act=a,
                               acte=ae,
                               intervene=i,
                               rew=r,
                               next_obs=s_,
                               done=d)
    

    def save_transition(self, output, timeend=0):
        self.replay_buffer.save_transitions(file='{}/{}'.format(output, timeend))

    def load_transition(self, output):
        if output is None: return
        self.replay_buffer.load_transitions('{}.npz'.format(output))
        
    
    def load_model(self, output):
        if output is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
    
    def load_actor(self, output):
        self.actor.load_state_dict(torch.load(output))
        
    def save_model(self, output):
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))
    
    def save_actor(self, output, no):
        torch.save(self.actor.state_dict(), '{}/actor{}.pkl'.format(output, no))
    
    def save(self, log_dir, epoch):
        state = {'actor':self.actor.state_dict(), 'actor_target':self.actor_target.state_dict(),
                 'actor_optimizer':self.actor_optimizer.state_dict(), 
                 'critic':self.critic.state_dict(), 'critic_target':self.critic_target.state_dict(),
                 'critic_optimizers':self.critic_optimizers.state_dict(),
                 'epoch':epoch}
        torch.save(state, log_dir)
        

    def load(self, log_dir):
        checkpoint = torch.load(log_dir)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic_optimizers.load_state_dict(checkpoint['critic_optimizers'])
    
