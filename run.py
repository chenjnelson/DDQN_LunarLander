#Import packages and dependencies
import random

import numpy as np
np.set_printoptions(precision=3)

import gym
from gym import wrappers

from agent.DDQN import DDQN

def train_models(num_ep, env_name):
  gammas = [.999,.99,.9]
  alphas = [.0001,.001,.01]

  test_model = False

  #Placeholder so that I don't run into a loop issue
  if test_model:
      gamma = [.1]
      alpha = [.1]

  for alpha in alphas:
      for gamma in gammas:

          #Take a snapshot of the environment
          env = gym.make(env_name)

          num_states = env.observation_space.shape[0]
          num_actions = env.action_space.n

          agent = DDQN(num_states, num_actions, gamma, alpha)

          #  Keep track of the current reward
          reward_avg = []

          if test_model:

              num_episodes = 100

              #No course of random action when testing the model
              agent.curr_epsilon = 0

          #Iterate over episodes
          else:
              num_episodes = num_ep

          for curr_episode in range(num_episodes):

              episode_reward = 0
              state = np.reshape(env.reset(), [1, num_states])

              #How long each episode can last
              for time in range(10000):

                  #Analyze the state, query from learner and take appropriate action
                  state_prime, reward, complete, nothing = env.step(agent.actionChoice(state))

                  # Add to memory
                  if not test_model:
                      agent.memory.append((state, agent.actionChoice(state), reward, np.reshape(state_prime, [1, num_states]), complete))

                      #Perform experience replay when memory gets long enough
                      if len(agent.memory) > agent.minibatch:
                          agent.experience_replay()

                  # Keep track of activity
                  state = np.reshape(state_prime, [1, num_states])
                  episode_reward += reward

                  # If episode is done, exit loop
                  if complete:
                      if not test_model:
                          agent.target_model.set_weights(agent.model.get_weights())
                      break

              #Decay
              if agent.curr_epsilon > agent.epsilon_min:
                  agent.curr_epsilon *= agent.decay

              #Append the reward
              reward_avg.append(episode_reward)

          env.close()

          #Store the results into a txt file for safekeeping
          with open('./results/rewards_avg'  + '_gamma_' + str(agent.gamma) + 'alpha_' + str(agent.alpha) +'.txt', 'a') as f:
              for x in reward_avg:
                  f.write(str(x) + '\n')

          agent.model.save_weights('./results/weights'  + '_gamma_' + str(agent.gamma) + 'alpha_' + str(agent.alpha) +'.h5')
  
def evaluate_model(gamma, alpha, model_name, env_name):
  test_model = True

  #Placeholder so that I don't run into a loop issue
  if test_model:
      gamma = [.1]
      alpha = [.1]

  for alpha in alpha:
      for gamma in gamma:

          #Take a snapshot of the environment
          env = gym.make(env_name)

          num_states = env.observation_space.shape[0]
          num_actions = env.action_space.n

          agent = DDQN(num_states, num_actions, gamma, alpha)

          #  Keep track of the current reward
          reward_avg = []

          if test_model:

              num_episodes = 100

              #No course of random action when testing the model
              agent.curr_epsilon = 0

              #Comment/Uncomment to run
              agent.model.load_weights(model_name)
              agent.alpha = alpha
              agent.gamma = gamma    

          else:
              num_episodes = 100

          for curr_episode in range(num_episodes):

              episode_reward = 0
              state = np.reshape(env.reset(), [1, num_states])

              for time in range(10000):

                  #Analyze the state, query from learner and take appropriate action
                  state_prime, reward, complete, nothing = env.step(agent.actionChoice(state))

                  # Add to memory
                  if not test_model:
                      agent.memory.append((state, agent.actionChoice(state), reward, np.reshape(state_prime, [1, num_states]), complete))

                      #Perform experience replay when memory gets long enough
                      if len(agent.memory) > agent.minibatch:
                          agent.experience_replay()

                  # Keep track of activity
                  state = np.reshape(state_prime, [1, num_states])
                  episode_reward += reward

                  # If episode is done, exit loop
                  if complete:
                      if not test_model:
                          agent.target_model.set_weights(agent.model.get_weights())
                      break

              #Decay
              if agent.curr_epsilon > agent.epsilon_min:
                  agent.curr_epsilon *= agent.decay

              #Append the reward
              reward_avg.append(episode_reward)

          env.close()

          #Store the results into a txt file for safekeeping
          with open('OUTCOME_rewards_avg'  + '_gamma_' + str(agent.gamma) + 'alpha_' + str(agent.alpha) +'.txt', 'a') as f:
              for x in reward_avg:
                  f.write(str(x) + '\n')

          #exit
          break

if __name__ == '__main__':

  num_episodes = 10

  env_name = 'LunarLander-v2'

  #Change this to false if you just want to evaluate the model
  train = False

  if train:
    train_models(num_episodes,env_name)
  else:
    evaluate_model(.999,.0001,'./results/' + 'weights_gamma_0.999alpha_0.0001.h5',env_name)