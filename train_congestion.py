import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as scio
import os
import pygame
import signal
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter   
writer = SummaryWriter('./algo/checkpoints/log')

from env_congestion import Congestion
from utils import set_seed, get_path



def train_congestion_task():
    
    set_seed(args.seed)

    
    # construct the DRL agent
    if args.algorithm == 0:
        from algo.TD3PHIL import DRL
        log_dir = 'algo/checkpoints/TD3PHIL.pth'
    elif args.algorithm == 1:
        from algo.TD3IARL import DRL
        log_dir = 'algo/checkpoints/TD3IARL.pth'
    elif args.algorithm == 2:
        from algo.TD3 import DRL
        log_dir = 'algo/checkpoints/TD3HIRL.pth'    
    elif args.algorithm == 3:
        from algo.TD3 import DRL    
        log_dir = 'algo/checkpoints/TD3.pth'
    
    
    # import CARLA environment
    env = Congestion(joystick_enabled = args.joystick_enabled, 
                   frame=args.simulator_render_frequency, port=args.simulator_port)


    
    s_dim = [env.observation_size_width, env.observation_size_height]
    a_dim = env.action_size
    
    DRL = DRL(a_dim, s_dim)
    
    exploration_rate = args.initial_exploration_rate 
    
    
    if args.resume and os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        DRL.load(log_dir)
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    
    
    if args.human_model:
        from algo.SL import SL
        SL = SL(a_dim, s_dim)
        SL.load_model('./algo/models_congestion/SL0.pkl')
    
    
    # initialize measurable variables
    total_step = 0
    a_loss,c_loss = 0,0
    count_sl = 0
    
    loss_critic, loss_actor = [], []
    episode_reward_list, global_reward_list, episode_duration_list = [], [], []
    
    drl_action = [[] for i in range(args.maximum_episode)]  # drl_action: output by RL policy
    adopted_action = [[] for i in range(args.maximum_episode)]  # adopted_action: eventual action which may include guidance action
    
    # reward_i: virtual reward (added shaping term); reward_e: real reward
    reward_i_record = [[] for i in range(args.maximum_episode)]
    reward_e_record = [[] for i in range(args.maximum_episode)]

    
    path_generator = get_path()

    
    start_time = time.perf_counter()
    
    for i in range(start_epoch, args.maximum_episode):
        
        # initial episode variables
        ep_reward = 0      
        step = 0
        step_intervened = 0
        done = False
        human_model_activated = False
        pid_intergal = 0
    
        # initial environment
        observation = env.reset()
        state = np.repeat(np.expand_dims(observation,2), 2, axis=2)
        state_ = state.copy()
        
    
        while not done:
            ## Section DRL's actting ##
            action = DRL.choose_action(state)
            action = np.clip( np.random.normal(action, exploration_rate), -1, 1)

            drl_action[i].append(action)
            ## End of Section DRL's actting
    
    
            ## Section human model's actting (actually achieved by a PI controller) ##
            # calculate some indicators of the environment to determine the activation of the PI controller
            ego_x = env.ego_vehicle.get_location().x
            ego_y = env.ego_vehicle.get_location().y
            ego_yaw = env.ego_vehicle.get_transform().rotation.yaw - 90
            
            central_line = path_generator(ego_y) # a reference trajectory's central line in x axis

            left_risk = ((ego_x > central_line+0.7) and (ego_yaw < 0)) or abs(ego_yaw)>6
            right_risk = ((ego_x < central_line-0.7) and (ego_yaw > 0)) or abs(ego_yaw)>6

            if not human_model_activated:
                pid_intergal = 0
            
            # you can use a pre-trained neural network model that mimic human behaviors or use a predefined PI controller as the human-guidance-model 
            # the activation condition of the human guidance model used in our paper:
            # the guidance is only actiated in the first 100 episodes and act per 5 episodes
            # and when the ego agent is approaching the left/right lane 
            if args.human_model:
                if (left_risk or right_risk) and (step != 0) and (i%5==1) and (i<100):
                    if args.human_model_type == 'PI':
                        human_model_activated = True
                        pid_intergal += (ego_x - central_line)
                        pid_proportional = ego_x - central_line
                        # use the noise-added output of the PI controller as the human guidance action
                        action = np.clip(0.2 * (pid_proportional) + 0.002 * (pid_intergal) + np.random.rand()*0.5-0.25, -1, 1)
                    else:
                        action = SL.choose_action(observation)[0]
                    env.show_human_model_mode()
                else:
                    human_model_activated = False
            ## End of Section pid controller ##
    
    
            ## Section environment update ##
            observation_, action_fdbk, reward_e, _, done, scope = env.step(action)
            
            state_[:,:,0] = state_[:,:,1].copy()
            state_[:,:,1] = observation_.copy()
            ## End of Section environment update ##
    
    
            ## Section reward shaping ##
            # intervention penalty-based shaping
            if args.reward_shaping == 1:
                # only the 1st intervened time step is penalized
                if (action_fdbk is not None) or (human_model_activated is True):
                    if step_intervened == 0:
                        reward_i = -10
                        step_intervened += 1
                    else:
                        reward_i = 0
                else:
                    reward_i = 0
                    step_intervened = 0
            # no shaping
            else:
                reward_i = 0
            reward = reward_e + reward_i
            ## End of Section reward shaping ##
    
    
            ## Section DRL store ##
            # human intervention event occurs
            if action_fdbk is not None:
    
                action = action_fdbk
                intervention = 1
                DRL.store_transition(state, action, action_fdbk, intervention, reward, state_)
            
                if args.human_model_update:
                    SL.store_transition(observation, action_fdbk)
                    count_sl += 1
            
            
            # human model intervention event occurs
            elif human_model_activated is True:
                intervention = 1
                action_fdbk = action
                DRL.store_transition(state, action, action_fdbk, intervention, reward, state_)
            
            
            # no intervention occurs
            else:
                intervention = 0
                DRL.store_transition(state, action, action_fdbk, intervention, reward, state_)
            ## End of DRL store ##
    
    
            ## Section DRL update ##
            learn_threshold = args.warmup_threshold if args.warmup else 256
            if total_step > learn_threshold:
                c_loss, a_loss = DRL.learn(batch_size = 128, epoch=i)
                loss_critic.append(np.average(c_loss))
                loss_actor.append(np.average(a_loss))
                # Decrease the exploration rate
                exploration_rate = exploration_rate * args.exploration_decay_rate if exploration_rate>args.cutoff_exploration_rate else args.cutoff_exploration_rate
            ## End of Section DRL update ##
    
    
            if args.human_model and args.human_model_update and count_sl > 100 and step % 10 == 0:
                SL.learn(batch_size = 32, epoch = i)
            
            
            ep_reward += reward
            global_reward_list.append([reward_e,reward_i])
            reward_e_record[i].append(reward_e)
            reward_i_record[i].append(reward_i)
            
            adopted_action[i].append(action)
    
            observation = observation_.copy()
            state = state_.copy()
    
            total_step += 1
            step += 1

            signal.signal(signal.SIGINT, env.signal_handler)
    
    
        # plt.plot(reward_e_record[i])
        # plt.plot(reward_i_record[i])
        # plt.show()
        
        mean_reward =  ep_reward / step  
        episode_reward_list.append(mean_reward)
        episode_duration_list.append(ego_y)
    
        print('\n episode is:',i)
        print('explore_rate:',round(exploration_rate,4))
        print('c_loss:',round(np.average(c_loss),4))
        print('a_loss',round(np.average(a_loss),4))
        print('total_step:',total_step)
        print('episode_step:',step)
        
        writer.add_scalar('reward/reward_episode', mean_reward, i)
        writer.add_scalar('reward/reward_episode_noshaping', np.mean(reward_e_record[i]), i)
        writer.add_scalar('reward/duration_episode', step, i)
        writer.add_scalar('rate_exploration', round(exploration_rate,4), i)
        writer.add_scalar('loss/loss_critic', round(np.average(c_loss),4), i)
        writer.add_scalar('loss/loss_actor', round(np.average(a_loss),4), i)
        

    
    print('total time:',time.perf_counter()-start_time)        
    
    timeend = round(time.time())
    DRL.save_actor('./algo/models_jam', timeend)

    
    pygame.display.quit()
    pygame.quit()
    
    action_drl = drl_action[0:i]
    action_final = adopted_action[0:i]
    scio.savemat('datacongestion{}-{}-{}.mat'.format(args.seed,args.algorithm,timeend), mdict={'action_drl': action_drl,'action_final': action_final,
                                                    'stepreward':global_reward_list,
                                                    'step':episode_duration_list,'reward':episode_reward_list,
                                                    'r_i':reward_i_record,'r_e':reward_e_record})
    



if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--algorithm', type=int, help='RL algorithm (0 for Proposed, 1 for IARL, 2 for HIRL, 3 for Vanilla TD3) (default: 0)', default=0)
    parser.add_argument('--human_model', action="store_true", help='whehther to use human behavior model (default: False)', default=False)
    parser.add_argument('--human_model_type', type=str, help='determine using a neural network (NN) or a PI controller (PI) as human guidance model (default: PI)', default='PI')
    parser.add_argument('--human_model_update', action="store_true", help='whehther to update NN human guidance model (default: False)', default=False)
    parser.add_argument('--maximum_episode', type=float, help='maximum training episode number (default:400)', default=400)
    parser.add_argument('--seed', type=int, help='fix random seed', default=2)
    parser.add_argument("--initial_exploration_rate", type=float, help="initial explore policy variance (default: 0.5)", default=0.5)
    parser.add_argument("--cutoff_exploration_rate", type=float, help="minimum explore policy variance (default: 0.05)", default=0.05)
    parser.add_argument("--exploration_decay_rate", type=float, help="decay factor of explore policy variance (default: 0.99988)", default=0.99988)
    parser.add_argument('--resume', action="store_true", help='whether to resume trained agents (default: False)', default=False)
    parser.add_argument('--warmup', action="store_true", help='whether to start training until collecting enough data (default: False)', default=False)
    parser.add_argument('--warmup_threshold', type=int, help='warmup length by step (default: 1e4)', default=1e4)
    parser.add_argument('--reward_shaping', type=int, help='reward shaping scheme (0: none; 1:proposed) (default: 1)', default=1)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--simulator_port', type=int, help='Carla port value which needs specifize when using multiple CARLA clients (default: 2000)', default=2000)
    parser.add_argument('--simulator_render_frequency', type=int, help='Carla rendering frequenze, Hz (default: 12)', default=12)
    parser.add_argument('--simulator_conservative_surrounding', action="store_true", help='surrounding vehicles are conservative or not (default: False)', default=False)
    parser.add_argument('--joystick_enabled', action="store_true", help='whether use Logitech G29 joystick for human guidance (default: False)', default=False)
    args = parser.parse_args()

    # Run
    train_congestion_task()
