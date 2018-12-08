import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
%matplotlib inline

from unityagents import UnityEnvironment
import numpy as np

from ounoise import OUNoise
from model import Actor, Critic
from replay_buffer import ReplayBuffer
from utils import create_agent, plot_scores

env = UnityEnvironment(file_name='agent_env/Reacher', no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = 100000  # replay buffer size 10^6 (paper)
BATCH_SIZE = 32        # minibatch size 64 (paper)
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 1e-4 (paper)
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay 1e-2 (paper)

agent1 = create_agent(env, brain_name, device, actor_file=None, critic_file=None, random_seed=39, lr_actor=LR_ACTOR,
                           lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, buffer_size=BUFFER_SIZE,
                           batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU)
agent2 = create_agent(env, brain_name, device, actor_file=None, critic_file=None, random_seed=39, lr_actor=LR_ACTOR,
                           lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, buffer_size=BUFFER_SIZE,
                           batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU)


def ddpg(n_episodes=5000, print_every=100):
    agent1_reward = []
    agent2_reward = []

    for i_episode in range(1, n_episodes + 1):
        # Reset environment
        env_info = env.reset(train_mode=True)[brain_name]
        # Reset agent
        agent1.reset()
        agent2.reset()
        # Get the initial state
        states = env_info.vector_observations

        episode_reward = np.array([0., 0.])

        while True:
            # Get action
            action_agent1 = agent1.act(states[0])
            action_agent2 = agent2.act(states[1])
            actions = np.array([action_agent1[0], action_agent2[0]])

            # Observe reaction (environment)
            env_info = env.step(actions)[brain_name]
            ## Get new state
            next_states = env_info.vector_observations
            ## Get reward
            rewards = env_info.rewards
            # See if episode has finished
            dones = env_info.local_done
            # Step
            agent1.step(states[0], action_agent1, rewards[0], next_states[0], dones[0])
            agent2.step(states[1], action_agent2, rewards[1], next_states[1], dones[1])

            states = next_states
            episode_reward += rewards

            if np.any(dones):
                break

        agent1_reward.append(episode_reward[0])
        agent2_reward.append(episode_reward[1])

        if i_episode % print_every == 0:
            avg_rewards = [np.mean(agent1_reward[-100:]), np.mean(agent2_reward[-100:])]
            print("\rEpisode {} - \tAverage Score: {:.2f} {:.2f}".format(i_episode, avg_rewards[0], avg_rewards[1]),
                  end="")

            torch.save(agent1.actor_local.state_dict(), 'agent1_actor_checkpoint.pth')
            torch.save(agent1.critic_local.state_dict(), 'agent1_critic_checkpoint.pth')

            torch.save(agent2.actor_local.state_dict(), 'agent2_actor_checkpoint.pth')
            torch.save(agent2.critic_local.state_dict(), 'agent2_critic_checkpoint.pth')

    return {'agent1_scores': agent1_reward, 'agent2_scores': agent2_reward}

scores = ddpg()

env.close()

plot_scores(scores['agent1_scores'])
plot_scores(scores['agent2_scores'])

max_scores = [max(scores['agent1_scores'][i], scores['agent2_scores'][i]) for i in range(len(scores['agent1_scores']))]
