from Game import Game
from DQNAgent import DQNAgent
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from common.utils import mini_batch_train

def main():
    env = Game()
    env.start()
    agent = DQNAgent(env)
    state = torch.from_numpy(np.zeros((4, 160, 240)))
    i = 0
    MAX_EPISODES = 100
    MAX_STEPS = 500
    BATCH_SIZE = 20
    episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)

if __name__ == '__main__':
    main()
