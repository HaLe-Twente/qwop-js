from Game import Game
from DQNAgent import DQNAgent
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    env = Game()
    env.start()
    agent = DQNAgent(env)
    state = torch.from_numpy(np.zeros((4, 160, 240)))
    i = 0
    MAX_ESPISODES = 20
    episodes = []
    scores = []

    # episodes = np.arange(500)
    # scores = np.random.randn(1, 500)

    while i <= MAX_ESPISODES:
        action = agent.get_action(state)

        observation, reward, done = env.step(action)

        prev_state = state
        state = torch.cat((state[1:], torch.from_numpy(np.array([observation]))), axis=0)
        agent.update_buffer(prev_state, action, reward, state, done)

        if done:
            i += 1
            episodes.append(i)
            scores.append(env.result)
            agent.update(batch_size=20)
            env.reset()

    print('Episodes')
    print(episodes)
    print('Scores:')
    print(scores)
    plt.scatter(episodes, scores, s=1)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Deep Q-Learning Agent')
    plt.savefig('score-500-episodes.png')
    # plt.show()

if __name__ == '__main__':
    main()
