import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from common.replay_buffers import BasicBuffer
from Model import ConvDQN, DQN

class DQNAgent:

    def __init__(self, env, use_conv=True, learning_rate=3e-4, gamma=0.99, buffer_size=10000, resume=False):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.0005

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_conv = use_conv
        if self.use_conv:
            self.path = 'saved-models/qwop_cnn.game.model'
            self.model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)

        else:
            self.path = 'saved-models/qwop_nn.game.model'
            self.model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)

        if resume:
            self.model.load_state_dict(torch.load(self.path))
            self.epsilon = 0.001
            self.model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.2):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1-self.epsilon_decay)
        if(np.random.randn() < self.epsilon):
            return self.env.action_space.sample()
        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.model.state_dict(), self.path)