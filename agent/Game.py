from Agent import Agent
from GameEnv import ActionSpace, ObservationSpace
import numpy as np
import logging
import time
import pytesseract
import cv2
import mss
import re
from datetime import date

WIDTH = 640
HEIGT = 400
SPACE = 64000

class Game:
    def __init__(self):
        self.agent = Agent()
        self.game_steps = 0
        self.action_space = ActionSpace(9)
        self.observation_space = ObservationSpace(SPACE) # 640*400*0.25
        self.old_score = 0
        self.result = 0

    def start(self):
        self.agent.start_game()
        self.game_steps = 0
        self.old_score = 0
        return self.step(4) # n = '4'

    def reload(self):
        self.game_steps = 0
        self.old_score = 0
        self.agent.hard_reload()
        self.agent.start_game()
        return self.step(4)

    def reset(self):
        self.game_steps = 0
        self.old_score = 0
        self.agent.hard_reload()
        self.agent.start_game()
        return self.step(4)

    def soft_reload(self):
        self.game_steps = 0
        self.old_score = 0
        self.agent.reload()

    def step(self, action):
        self.agent.start_game()
        self.game_steps += 1
        self.agent.step(action)
        shot = self.get_screen_shot()
        done = self.is_done(shot)
        distance_score = self.get_score() or 0
        time_score = - (self.game_steps/(abs(distance_score)+1e5)) # The higher the pace, the slowest it goes
        score = distance_score -  self.old_score
        self.old_score = distance_score
        if done:
            self.soft_reload()
            score = -100
            self.result = distance_score
        return shot, score, done

    def is_done(self, shot):
        return self.agent.game_ended()

    def get_score(self):
        return self.agent.get_score()

    def get_screen_shot_timed(self):
        start = time.time()
        img = self.get_screen_shot()
        print(time.time() - start)
        return img

    def get_screen_shot(self, render = False):
        with mss.mss() as sct:
            monitor = {"top": 175, "left": 20, "width": WIDTH, "height": HEIGT}
            shot = sct.grab(monitor)
            img = np.array(shot)
            img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_rgb_resized = cv2.resize(img_grey, (240, 160), interpolation=cv2.INTER_CUBIC)
            if render: self.render(img)
        return img_rgb_resized

    def render(self, img):
        cv2.waitKey(1)
