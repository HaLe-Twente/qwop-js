import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import math
import gym
import time
from datetime import datetime
import matplotlib.pyplot as plt

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    episodes = []
    scores = []
    state = torch.from_numpy(np.zeros((4, 160, 240)))
    f = open('best_record.txt')
    best_record = float(f.readline())
    f.close()

    for episode in range(max_episodes):
        episode_reward = 0
        start_time = datetime.now()
        for step in range(max_steps):
            action = agent.get_action(state)
            observation, reward, done = env.step(action)

            prev_state = state
            state = torch.cat((state[1:], torch.from_numpy(np.array([observation]))), axis=0)
            agent.update_buffer(prev_state, action, reward, state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps-1:
                end_time = datetime.now()
                episodes.append(episode)
                score = env.result
                scores.append(score)
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": Reward = " + str(episode_reward))
                print("Episode " + str(episode) + ": Score = " + str(score))
                print("Episode " + str(episode) + ": Epsilon = " + str(agent.epsilon))
                if score >= 100:
                    finish_time = (end_time - start_time).total_seconds()
                    if finish_time < best_record:
                        best_record = finish_time
                break

        if episode % 10 == 0:
            agent.save_model()

    date = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
    plt.scatter(episodes, scores, s=1)
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.xticks(np.arange(0, max_episodes, 1), np.arange(0, max_episodes, 1))
    plt.title('Deep Q-Learning Agent')
    plt.savefig('rewards-episodes'+ date +'.png')
    f = open('best_record.txt', "w")
    f.write(str(best_record))
    f.close()

    return episode_rewards
