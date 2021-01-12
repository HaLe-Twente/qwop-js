from agent.Game import Game
import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    game = Game()
    game.start()

    while True:
        observation, reward, done = game.execute_action("r")
        #print(reward)
        if done:
            plt.imshow(observation)
            plt.show()
            print(f'reward {reward}')
            game.reload()

if __name__ == '__main__':
    main()