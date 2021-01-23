from agent.Game import Game
from agent.DQNAgent import DQNAgent
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    env = Game()
    env.start()
    agent = DQNAgent(env)
    state = torch.from_numpy(np.zeros((4, 160, 240)))
    exp_set = set()


    while True:
        action = agent.get_action(state)
        observation, reward, done = env.execute_action(action)

        prev_state = state
        state = torch.cat((state[1:], torch.from_numpy(np.array([observation]))), axis=0)
        experience = (prev_state, torch.from_numpy(np.array(action)), reward, state, done)
        exp_set.add(experience)
        agent.update_buffer(prev_state, action, reward, state, done)

        if done:
            exp_set = set()
            agent.update(batch_size=20)
            env.reset()

if __name__ == '__main__':
    main()