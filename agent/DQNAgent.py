import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from agent.ReplayBuffer import ReplayBuffer
#from Model import ConvDQN, DQN
from agent.ConvDQN import ConvDQN

class DQNAgent:

    def __init__(self, env, use_conv=True, learning_rate=3e-4, gamma=0.99, buffer_size=10000, resume=False):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.0005

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_conv = use_conv
        if self.use_conv:
            self.path = 'saved-models/qwop_cnn.game.model'
            self.model = ConvDQN() #ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)

        else:
            self.path = 'saved-models/qwop_nn.game.model'
            #self.model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)

        if resume:
            self.model.load_state_dict(torch.load(self.path))
            self.epsilon = 0.01
            self.model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.2):
        state = torch.unsqueeze(state, 0).float().to(self.device)
        #state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = qvals.tolist()[0]#np.argmax(qvals.cpu().detach().numpy())


        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1-self.epsilon_decay)
        if(np.random.randn() < self.epsilon):
            action = self.env.action_space.sample()
        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        #states = torch.FloatTensor(states).to(self.device)
        #actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        #next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q = []
        next_Q = []
        for old, new in states, next_states:
            curr_Q_state = self.model.forward(torch.unsqueeze(old, 0).float())#.gather(1, actions.unsqueeze(1))
            next_Q_state = self.model.forward(torch.unsqueeze(new, 0).float())
            curr_Q.append(curr_Q_state.tolist())
            next_Q.append(next_Q_state.tolist())

        curr_Q = torch.FloatTensor(curr_Q)
        next_Q = torch.FloatTensor(next_Q)
        print("curr:", curr_Q)
        print("action:", actions)
        curr_Q = curr_Q.squeeze(1)
        curr_Q = curr_Q.gather(1, actions.unsqueeze(1))
        next_Q = next_Q.squeeze(1)
        max_next_Q = torch.max(next_Q, 1)[0]

        expected_Q = self.gamma * max_next_Q + rewards#.squeeze(1)

        print("curr:", curr_Q)
        print("next:", next_Q)
        print("max:", max_next_Q)
        print("exp:", expected_Q)
        curr_Q.requires_grad_()
        expected_Q.requires_grad_()

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        try:
            batch = self.replay_buffer.sample(batch_size)
            loss = self.compute_loss(batch)
            print("loss: ", loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        except ValueError:
            pass

    def update_buffer(self, prev_state, action, reward, next_state, done):
        self.replay_buffer.push(prev_state, action, reward, next_state, done)

    def save_model(self):
        torch.save(self.model.state_dict(), self.path)