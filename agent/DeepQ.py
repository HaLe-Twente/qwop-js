from agent.Game import Game
from agent.ReplayBuffer import ReplayBuffer
from agent.ConvDNQ import ConvDNQ

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd


import random


class DNQAgent:
    def __init__(self, env, net):
        self.env = env
        self.ROWS = 160
        self.COLS = 240
        self.REM_STEP = 4
        self.EPISODES = 10
        self.image_memory = torch.from_numpy(np.zeros((self.REM_STEP, self.ROWS, self.COLS)))
        self.buffer = ReplayBuffer(100000)
        self.net = net

    def init_weights(self):
        if type(self.net) == nn.Conv2d or type(self.net) == nn.Linear:
            torch.nn.init.uniform(self.net.weight, -0.01, 0.01)
            self.net.bias.data.fill_(0.01)

    def getimage(self):
        return self.env.get_screen_shot()

    def update_image_memory(self, image):
        self.image_memory = torch.cat((self.image_memory[1:], torch.from_numpy(np.array([image]))), axis=0)

    def reset(self):
        self.env.reset()
        for i in range(self.REM_STEP):
            state = self.getimage()
        return state

    def getaction(self):
        input_conv = torch.unsqueeze(self.image_memory, 0).float()
        action = self.net(input_conv)
        return torch.argmax(action).item()

    def step(self, action):
        next_state, reward, done = self.env.execute_action(action)
        next_state = self.getimage()
        return next_state, reward, done

    def train(self, sample_size):
        #try:
            batch = (self.buffer.sample(sample_size))
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
            print("works1")
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    next_state_batch)), dtype=torch.bool)
            print("works2")
            non_final_next_states = torch.cat([s for s in next_state_batch
                                               if s is not None])
            print("works3")
            state_action_values = []
            for state in state_batch:
                state_action_values.append(self.net(torch.unsqueeze(state, 0).float()))#.gather(1, action_batch))

            print("works4")
            q_value = torch.sum(state_action_values * action_batch, dim=1)
            print("works5")




        #except ValueError:
            #print(ValueError)
            #print("not enough memory")
            #pass

    def run(self):
        # Each of this episode is its own game.
        optimizer = optim.Adam(self.net.parameters())
        self.env.start()
        for episode in range(self.EPISODES):
            self.reset()
            # this is each frame, up to 500...but we wont make it that far with random.
            for t in range(500):
                # This will just create a sample action in any environment.
                # In this environment, the action can be 0 or 1, which is left or right
                action = self.getaction()  # this executes the environment with an action,

                # and returns the observation of the environment,
                # the reward, if the env is over, and other info.
                old_dist = 0
                old_memory = self.image_memory
                next_state, dist, done = self.step(self.env.action_space[action])
                reward = dist-old_dist
                old_dist = dist
                self.update_image_memory(next_state)
                self.buffer.push(old_memory, action, reward, self.image_memory, done)
                #print(reward)
                #print("score: ", self.env.agent.get_score())



                # lets print everything in one line:
                # print(t, next_state, reward, done, info, action)
                if done:
                    break
            #self.train(2)




def main():
    game = Game()
    net = ConvDNQ()#.float()
    agent = DNQAgent(game, net)
    agent.run()








if __name__ == '__main__':
    main()