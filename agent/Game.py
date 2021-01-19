from agent.Agent import Agent
from GameEnv import ActionSpace, ObservationSpace
import numpy as np
import logging
import time
import pytesseract
import cv2
import mss
import re
import torch

class Game:
    def __init__(self):
        self.agent = Agent()
        self.game_steps = 0
        self.action_space = ActionSpace(9)
        #self.observation_space = ObservationSpace(SPACE)  # 640*400*0.25

    def start(self):
        self.agent.start_game()
        self.game_steps = 0
        return self.execute_action(5)#'n')

    def reset(self):
        self.game_steps = 0
        self.agent.hard_reload()
        self.agent.start_game()
        return self.execute_action(5)#'n')

    def soft_reload(self):
        self.game_steps = 0
        self.agent.reload()

    def execute_action(self, action):
        self.agent.start_game()
        self.game_steps += 1
        #self.agent.unpause()
        #for char in action:
        #getattr(self.agent, action)()
        self.agent.step(action)
        shot = self.get_screen_shot()
        #self.agent.pause()
        done = self.is_done(shot)
        score = self.get_score()
        if done:
            distance_score = self.get_score()
            #time_score = - (self.game_steps/(abs(distance_score)+1e5)) # The higher the pace, the slowest it goes
            score = distance_score #+ time_score
            self.soft_reload()
        return shot, score, done

    def is_done(self, shot):
        # blueidx = shot[:, :] < 24
        # notblueidx = shot[:, :] >= 24
        # shot[blueidx] = 255
        # shot[notblueidx] = 0
        # np.savetxt('sample_shot', shot[15:20,66:])
        # mask = np.array([[0,0,255],[0,255,255],[0,255,255],[0,255,255],[0,0,255],[0,0,255]])
        # return np.array_equal(shot[15:21,66:], mask)
        # if self.agent.game_ended():
        #     print('ended')
        return self.agent.game_ended()

    def get_score(self):
        # with mss.mss() as sct:
        #     shot = sct.grab({"top": 155, "left": 140, "width": 350, "height": 70})
        #     img = ~(np.array(shot)[:,:,0]) #removes rgb and inverts colors
        #     img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1] #threshold to remove color artifacts and leave it black and white
        #     score = pytesseract.image_to_string(image=img)
        #     digits_rgx = re.compile("-?[0-9]+.?[0-9]")
        #     result = digits_rgx.findall(score)
        #     if len(result) > 0:
        #         score = result[0]
        #     else:
        #         score = 0
        # return float(score)
        #print(self.agent.get_score())
        return self.agent.get_score()

    def get_screen_shot_timed(self):
        start = time.time()
        img = self.get_screen_shot()
        print(time.time() - start)
        return img

    def get_screen_shot(self, render = False):
        with mss.mss() as sct:
            shot = sct.grab({"top": 170, "left": 90, "width": 550, "height": 550})
            """
            TODO:
            TEST ONLY GRAYSCALE
            this processing might not be useful since the important data is in the difference between frames
            """
            img = np.array(shot)
            #img[:, :, 2] = 0
            #img[:, :, 1] = 0
            #blueidx = img[:, :, 0] < 24
            #notblueidx = img[:, :, 0] >= 24
            #img[blueidx] = 255
            #img[notblueidx] = 0
            img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_rgb_resized = cv2.resize(img_grey, (240, 160), interpolation=cv2.INTER_CUBIC)


            #img = np.array(shot)[:,:,0]
            #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            if render: self.render(img)
        return img_rgb_resized

    def render(self, img):
        cv2.imshow('window', img)
        cv2.waitKey(1)
