# load the weights from file
from unityagents import UnityEnvironment
import numpy as np
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import os

dirpath = os.path.dirname(os.path.abspath(__file__))

env = UnityEnvironment(file_name=dirpath + "/../Banana.app")

max_t = 1000
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

agent = Agent(state_size=state_size, action_size=action_size)
agent.qnetwork_local.load_state_dict(
    torch.load(dirpath + "/checkpoint.pth"))
max_t = 1000
for i in range(10):
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]
    score = 0
    for t in range(max_t):
        action = agent.act(state, 0.01)
        # send the action to the environment
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has
        score += reward
        state = next_state
        if done:
            break
    print(score)

env.close()


# from unityagents import UnityEnvironment
# import numpy as np
# import gym
# import random
# import torch
# import numpy as np
# from collections import deque
# import matplotlib.pyplot as plt
# from dqn_agent import Agent
# import pickle
# import os
#
# dirpath = os.path.dirname(os.path.abspath(__file__))
#
# env = UnityEnvironment(file_name=dirpath + "/../Banana.app")
#
# # get the default brain
# brain_name = env.brain_names[0]
# brain = env.brains[brain_name]
#
# # reset the environment
# env_info = env.reset(train_mode=False)[brain_name]
#
# # number of agents in the environment
# print('Number of agents:', len(env_info.agents))
#
# # number of actions
# action_size = brain.vector_action_space_size
# print('Number of actions:', action_size)
#
# # examine the state space
# state = env_info.vector_observations[0]
# print('States look like:', state)
# state_size = len(state)
# print('States have length:', state_size)
#
# agent = Agent(state_size=state_size, action_size=action_size)
# agent.qnetwork_local.load_state_dict(torch.load(dirpath + '/checkpoint.pth'))
#
#
# def dqn(n_episodes=2000, max_t=1000):
#     """Deep Q-Learning.
#
#     Params
#     ======
#         n_episodes (int): maximum number of training episodes
#         max_t (int): maximum number of timesteps per episode
#         eps_start (float): starting value of epsilon, for epsilon-greedy action selection
#         eps_end (float): minimum value of epsilon
#         eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
#     """
#     scores = []                        # list containing scores from each episode
#     scores_window = deque(maxlen=100)  # last 100 scores
#     eps = 0.0                  # initialize epsilon
#     for i_episode in range(1, n_episodes + 1):
#         env_info = env.reset(train_mode=True)[
#             brain_name]  # reset the environment
#         state = env_info.vector_observations[0]
#         score = 0
#         for t in range(max_t):
#             action = agent.act(state, eps)
#
#             # send the action to the environment
#             env_info = env.step(action)[brain_name]
#             next_state = env_info.vector_observations[0]   # get the next state
#             reward = env_info.rewards[0]                   # get the reward
#             # see if episode has finished
#             done = env_info.local_done[0]
#
#             agent.step(state, action, reward, next_state, done)
#             state = next_state
#             score += reward
#
#             if done:
#                 break
#
#         scores_window.append(score)       # save most recent score
#         scores.append(score)              # save most recent score
#
#         print('\rEpisode {}\tAverage Score: {:.2f} \t Epsilon: {}'.format(
#             i_episode, np.mean(scores_window), eps), end="")
#         if i_episode % 100 == 0:
#             print('\rEpisode {}\tAverage Score: {:.2f} \t Epsilon: {}'.format(
#                 i_episode, np.mean(scores_window), eps))
#         if np.mean(scores_window) >= 13.0:
#             print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
#                 i_episode, np.mean(scores_window)))
#             torch.save(agent.qnetwork_local.state_dict(),
#                        dirpath + '/checkpoint.pth')
#             break
#     return scores
#
#
# scores = dqn()
