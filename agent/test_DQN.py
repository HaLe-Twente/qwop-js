import gym
from Game import Game
from DQN import DQNAgent
from common.utils import mini_batch_train

# env_id = "CartPole-v0"
MAX_EPISODES = 100
MAX_STEPS = 500
BATCH_SIZE = 32

env = Game()
agent = DQNAgent(env, use_conv=False, resume=True)
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)
