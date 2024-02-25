from tqdm import tqdm
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import gymnasium as gym
import random
from time import time
import matplotlib.pyplot as plt

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

# def greedy_action(Q,s,nb_actions):
#     Qsa = []
#     for a in range(nb_actions):
#         sa = np.append(s,a).reshape(1, -1)
#         Qsa.append(Q.predict(sa))
#     best_a = np.argmax(Qsa)
#     return best_a


# def rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
#     nb_samples = S.shape[0]
#     Qfunctions = []
#     SA = np.append(S,A,axis=1)
#     for iter in tqdm(range(iterations), disable=disable_tqdm):
#         if iter==0:
#             value=R.copy()
#         else:
#             Q2 = np.zeros((nb_samples,nb_actions))
#             for a2 in range(nb_actions):
#                 A2 = a2*np.ones((S.shape[0],1))
#                 S2A2 = np.append(S2,A2,axis=1)
#                 Q2[:,a2] = Qfunctions[-1].predict(S2A2)
#             max_Q2 = np.max(Q2,axis=1)
#             value = R + gamma*(1-D)*max_Q2
#         Q = RandomForestRegressor()
#         Q.fit(SA,value)
#         Qfunctions.append(Q)
#     return Qfunctions


# def collect_samples(env, horizon, disable_tqdm=False, print_done_states=False):
#     s, _ = env.reset()
#     S = []
#     A = []
#     R = []
#     S2 = []
#     D = []
#     for _ in tqdm(range(horizon), disable=disable_tqdm):
#         a = env.action_space.sample()
#         s2, r, done, trunc, _ = env.step(a)
#         S.append(s)
#         A.append(a)
#         R.append(r)
#         S2.append(s2)
#         D.append(done)
#         if done or trunc:
#             s, _ = env.reset()
#             if done and print_done_states:
#                 print("done!")
#         else:
#             s = s2
#     S = np.array(S)
#     A = np.array(A).reshape((-1,1))
#     R = np.array(R)
#     S2= np.array(S2)
#     D = np.array(D)
#     return S, A, R, S2, D

# class ProjectAgent:
#     def __init__(self, Qvalue=None) -> None:
#         self.Qvalue = Qvalue
#         self.nb_actions = 4

#     def act(self, observation, use_random=False):

#         action = greedy_action(self.Qvalue, observation, self.nb_actions)
#         return action

#     def save(self, path):
#         with open(path, "wb") as f:
#             pickle.dump(self.Qvalue, f)

#     def load(self):
#         with open("model.pkl", "rb") as f:
#             self.Qvalue = pickle.load(f)

# if __name__ == "__main__":
#     gamma = .95
#     nb_iter = 30
#     nb_actions = env.action_space.n
#     S,A,R,S2,D = collect_samples(env, int(10000))
#     Qfunctions = rf_fqi(S, A, R, S2, D, nb_iter, nb_actions, gamma)
#     Qvalue = Qfunctions[-1]
#     agent = ProjectAgent(Qvalue)
#     agent.save("model.pkl")




n_action = env.action_space.n 

# DQN config
config = {'nb_actions': n_action,
        'learning_rate': 0.001,
        'gamma': 0.95,
        'buffer_size': 1000000,
        'epsilon_min': 0.01,
        'epsilon_max': 1.,
        'epsilon_decay_period': 1000,
        'epsilon_delay_decay': 20,
        'batch_size': 20}

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

def greedy_action(network, state):
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to("cpu"))
        return torch.argmax(Q).item()

class ProjectAgent:
    def __init__(self, config=config, model=None):
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], "cpu")
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model
        if self.model is not None :
            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.nb_actions = 4

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        for episode in range(max_episode):
            t0 = time()
            for _ in range(50):
                # update epsilon
                if step > self.epsilon_delay:
                    epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

                # select epsilon-greedy action
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = greedy_action(self.model, state)

                # step
                next_state, reward, done, trunc, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                episode_cum_reward += reward

                # train
                self.gradient_step()

                state = next_state
                step += 1
                                
            print("Episode ", '{:3d}'.format(episode), 
                    ", epsilon ", '{:6.2f}'.format(epsilon), 
                    ", batch size ", '{:5d}'.format(len(self.memory)), 
                    ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                    ", execution time ", '{:4.1f}'.format(time()-t0),
                    ", step ", '{:5d}'.format(step),
                    sep='')
            state, _ = env.reset()
            episode_return.append(episode_cum_reward)
            episode_cum_reward = 0

        # while episode < max_episode:
        #     # update epsilon
        #     if step > self.epsilon_delay:
        #         epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

        #     # select epsilon-greedy action
        #     if np.random.rand() < epsilon:
        #         action = env.action_space.sample()
        #     else:
        #         action = greedy_action(self.model, state)

        #     # step
        #     next_state, reward, done, trunc, _ = env.step(action)
        #     self.memory.append(state, action, reward, next_state, done)
        #     episode_cum_reward += reward

        #     # train
        #     self.gradient_step()

        #     # next transition
        #     step += 1
        #     if done:
        #         episode += 1
        #         print("Episode ", '{:3d}'.format(episode), 
        #               ", epsilon ", '{:6.2f}'.format(epsilon), 
        #               ", batch size ", '{:5d}'.format(len(self.memory)), 
        #               ", episode return ", '{:4.1f}'.format(episode_cum_reward),
        #               sep='')
        #         state, _ = env.reset()
        #         episode_return.append(episode_cum_reward)
        #         episode_cum_reward = 0
        #     else:
        #         state = next_state

        return episode_return

    def act(self, observation, use_random=False):
        if use_random:
            action = np.random(self.nb_actions)
        else:
            action = greedy_action(self.model, observation)
        return action

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        with open("model.pkl", "rb") as f:
            self.model = pickle.load(f)
            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])


if __name__ == "__main__":
    # Declare network
    state_dim = env.observation_space.shape[0]
    nb_neurons=24
    DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                            nn.ReLU(),
                            nn.Linear(nb_neurons, nb_neurons),
                            nn.ReLU(), 
                            nn.Linear(nb_neurons, n_action)).to("cpu")

    
    # Train agent
    agent = ProjectAgent(config, DQN)
    scores = agent.train(env, 200)
    plt.plot(scores)
    plt.show()
    agent.save("model.pkl")



# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import trange
# from torch.distributions import Categorical

# class policyNetwork(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         state_dim = env.observation_space.shape[0]
#         n_action = env.action_space.n
#         self.fc1 = nn.Linear(state_dim, 128)
#         self.fc2 = nn.Linear(128, n_action)

#     def forward(self, x):
#         if x.dim() == 1:
#             x = x.unsqueeze(dim=0)
#         x = F.relu(self.fc1(x))
#         action_scores = self.fc2(x)
#         return F.softmax(action_scores,dim=1)

#     def sample_action(self, x):
#         probabilities = self.forward(x)
#         action_distribution = Categorical(probabilities)
#         return action_distribution.sample().item()

#     def log_prob(self, x, a):
#         probabilities = self.forward(x)
#         action_distribution = Categorical(probabilities)
#         return action_distribution.log_prob(a)

# class reinforce_agent:
    # def __init__(self, config, policy_network):
    #     self.device = "cpu"
    #     self.scalar_dtype = next(policy_network.parameters()).dtype
    #     self.policy = policy_network
    #     self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
    #     lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
    #     self.optimizer = torch.optim.Adam(list(self.policy.parameters()),lr=lr)
    #     self.nb_episodes = config['nb_episodes'] if 'nb_episodes' in config.keys() else 1
    
    # def one_gradient_step(self, env):
    #     # run trajectories until done
    #     episodes_sum_of_rewards = []
    #     states = []
    #     actions = []
    #     returns = []
    #     for ep in range(self.nb_episodes):
    #         x,_ = env.reset()
    #         rewards = []
    #         episode_cum_reward = 0
    #         while(True):
    #             a = self.policy.sample_action(torch.as_tensor(x))
    #             y,r,done,trunc,_ = env.step(a)
    #             states.append(x)
    #             actions.append(a)
    #             rewards.append(r)
    #             episode_cum_reward += r
    #             x=y
    #             if done: 
    #                 # The condition above should actually be "done or trunc" so that we 
    #                 # terminate the rollout also if trunc=True.
    #                 # But then, our return-to-go computation would be biased as we would 
    #                 # implicitly assume no rewards can be obtained after truncation, which 
    #                 # is wrong.
    #                 # We leave it as is for now (which means we will call .step() even 
    #                 # after trunc=True) and will discuss it later.
    #                 # Compute returns-to-go
    #                 new_returns = []
    #                 G_t = 0
    #                 for r in reversed(rewards):
    #                     G_t = r + self.gamma * G_t
    #                     new_returns.append(G_t)
    #                 new_returns = list(reversed(new_returns))
    #                 returns.extend(new_returns)
    #                 episodes_sum_of_rewards.append(episode_cum_reward)
    #                 break
    #     # make loss
    #     returns = torch.tensor(returns)
    #     log_prob = self.policy.log_prob(torch.as_tensor(np.array(states)),torch.as_tensor(np.array(actions)))
    #     loss = -(returns * log_prob).mean()
    #     # gradient step
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return np.mean(episodes_sum_of_rewards)

    # def train(self, env, nb_rollouts):
    #     avg_sum_rewards = []
    #     for ep in trange(nb_rollouts):
    #         avg_sum_rewards.append(self.one_gradient_step(env))
    #     return avg_sum_rewards