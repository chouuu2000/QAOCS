import os
import torch
import numpy as np
from env import ABREnv
from PPO import PPO
from core import S_INFO, S_LEN, A_DIM
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
def pmkdir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
def test(args):
    ####### initialize environment hyperparameters ######
    env_name = args.env_name
    checkpoint_path = args.checkpoint_path
    test_trace_folder = args.test_trace_folder
    video_path = args.video_path
    random_seed = args.random_seed
    test_episodes = args.test_episodes
    #####################################################

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 2000  # max timesteps in one episode


    writer = SummaryWriter()
    #####################################################
    Total_start_time = time.time()
    # Create tmp test trace directory to store one trace file and copy the trace file in it

    done_path = test_trace_folder + 'done/'
    if not os.path.exists(done_path):
        pmkdir(done_path)
    for trace in os.listdir(test_trace_folder):
        tmp_path = test_trace_folder + 'tmp/'
        if not os.path.exists(tmp_path): 
            pmkdir(tmp_path)
        
        if trace == 'tmp' or trace == 'done':
            continue
        trace_path = os.path.join(test_trace_folder, trace)
        print(f'Trace: {trace_path}')
        # Copy one trace file in tmp directory
        os.system(f'cp {trace_path} {tmp_path+trace}')
        print(f'Trace: {trace} copied to {tmp_path}')

        print("============================================================================================")
        print("Testing on trace: {}".format(tmp_path))
    
        env = ABREnv(trace_folder=tmp_path, video_path=video_path,\
                    random_seed=random_seed,writer=writer, test =True) 
        print("Environment created")
        # state space dimension
        state_dim = S_LEN * S_INFO

        # action space dimension
        action_dim = A_DIM
        action_std = 0.6  # starting std for action distribution (Multivariate Normal)
        action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
        action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
        #####################################################

        ## Note : print/log frequencies should be > than max_ep_len

        ################ PPO hyperparameters ################
        update_timestep = max_ep_len * 4  # update policy every n timesteps
        K_epochs = 80  # update policy for K epochs in one PPO update

        eps_clip = 0.2  # clip parameter for PPO
        gamma = 0.99  # discount factor
        
        lr_actor = 0.0003  # learning rate for actor network
        lr_critic = 0.001  # learning rate for critic network

        # initialize a PPO agent
        ppo_agent = PPO(state_dim, action_dim, 0, 0, 0, 0, 0, has_continuous_action_space, 0)
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                        action_std)
        # load the trained model checkpoint

        if os.path.exists(checkpoint_path):
            ppo_agent.load(checkpoint_path)
            print('checkpoint path:', checkpoint_path)
        # checkpoint = torch.load(checkpoint_path)
        # ppo_agent.policy.load_state_dict(checkpoint)

        print("--------------------------------------------------------------------------------------------")
        test_running_reward = 0

        for ep in range(1, test_episodes + 1):
            ep_reward = 0
            state = env.reset()
            state = state.flatten()
            test_time_start = time.time()
            for t in range(1, max_ep_len + 1):
                # select action with policy
                print(f'Timestep: {t}')
                action = ppo_agent.select_action(state)
                state, reward, done, _ = env.step(action)
                state = state.flatten()
                ep_reward += reward

                if done:
                    break

            # clear buffer
            test_time_end = time.time()
            ppo_agent.buffer.clear()
            test_running_reward += ep_reward
            print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
            print(f'Test Time: {test_time_end - test_time_start:.2f}s')
            ep_reward = 0

        env.close()
        
        print("--------------------------------------------------------------------------------------------")
        print("Trace: {} Simulation Done".format(trace) )
        print("Test Episodes: {} \t\t Average Test Reward: {}".format(test_episodes, round(test_running_reward / test_episodes, 2)))

        # Copy the trace file to done directory
        os.system(f'cp {trace_path} {done_path+trace}')
        # remove the trace file from the test directory
        os.system(f'rm -r {trace_path}')
        # remove the tmp directory
        os.system(f'rm -r {tmp_path}')
        print("Successfully removed the tmp trace directory")
        print("============================================================================================")
    Total_end_time = time.time()
    print(f'Total Testing Time: {Total_end_time - Total_start_time:.2f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="PPO_preTrained/ABREnv/PPO_ABREnv_0_2.pth")
    parser.add_argument("--test_trace_folder", type=str, default="./test/")
    parser.add_argument("--video_path", type=str, default="bigbuckbunny360p24.mp4")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--test_episodes", type=int, default=1)
    parser.add_argument("--env_name", type=str, default='ABREnv')
    parser.add_argument("--has_continuous_action_space", type=bool, default=True)
    args = parser.parse_args()
    test(args)