import os
from datetime import datetime
import torch
import numpy as np
from env import ABREnv
from PPO import PPO
from core import S_INFO, S_LEN, A_DIM
import time
from torch.utils.tensorboard import SummaryWriter
################################### Training ###################################
def train():
    ####### initialize environment hyperparameters ######
    env_name = "ABREnv"

    has_continuous_action_space = True  # continuous action space; else discrete

    # max_ep_len = 100000  # max timesteps in one episode
    max_ep_len = 1000
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len  # print avg reward in the interval (in num timesteps)
    log_freq =  max_ep_len  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(500)  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = 200*4  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.97  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################


    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    # tensorboard for logging
    # Create a TensorBoard summary writer
    writer = SummaryWriter()
    
    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    print("============================================================================================")



    print("training environment name : " + env_name)

    env = ABREnv(writer=writer)

    # state space dimension
    state_dim = S_LEN * S_INFO

    # action space dimension
    action_dim = A_DIM



    ################### checkpointing ###################
    run_num_pretrained = 2  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    # load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        ppo_agent.load(checkpoint_path)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    
    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 1
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    best_avg_reward = -np.inf
    num_episodes_without_improvement = 0
    max_episodes_without_improvement = 50
    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        state = state.flatten()
        current_ep_reward = 0

        ep_start = time.time()
        for t in range(1, max_ep_len + 1):
            print(f'Episode: {i_episode} ,Timestep: {t}')

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            print('reward:', reward)
            state = state.flatten()

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            writer.add_scalar('Timestep Reward', reward, time_step)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                policy_loss, value_loss,entropy_loss = ppo_agent.update()
                loss = policy_loss + value_loss + entropy_loss
                # Log individual scalar values
                # writer.add_scalar('Loss/Total_loss', loss.mean().item(), time_step)
                # writer.add_scalar('Loss/Policy_loss', policy_loss.mean().item(), time_step)
                # writer.add_scalar('Loss/Value_loss', value_loss.mean().item(), time_step)
                # writer.add_scalar('Loss/Entropy_loss', entropy_loss.mean().item(), time_step)
                writer.add_scalars('Loss',   {'Total loss':loss.mean().item(),'policy_loss': policy_loss.mean().item(), \
                                            'value_loss': value_loss.mean().item(), 'entropy_loss': entropy_loss.mean().item()}, time_step)
                                            


            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                if log_running_episodes > 0:
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)
                    ## Early stopping
                    if log_avg_reward > best_avg_reward:
                        best_avg_reward = log_avg_reward
                        num_episodes_without_improvement = 0
                    else:
                        num_episodes_without_improvement += log_running_episodes

                    if num_episodes_without_improvement > max_episodes_without_improvement:
                        print(f"Early stopping at episode {i_episode}")
                        break

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break
        ep_end = time.time()
        print(f'Episode{i_episode} take Time: {ep_end - ep_start:.2f}s')
        writer.add_scalar('Current Episode Reward', current_ep_reward, i_episode)
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()
    writer.close()
    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
