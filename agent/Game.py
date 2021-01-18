from Agent import Agent
from GameEnv import ActionSpace, ObservationSpace
import numpy as np
import logging
import time
import pytesseract
import cv2
import mss
import re

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

    # def execute_action(self, action):
    #     self.agent.start_game()
    #     self.game_steps += 1
    #     #self.agent.unpause()
    #     for char in action:
    #         getattr(self.agent, char)()
    #     shot = self.get_screen_shot()
    #     #self.agent.pause()
    #     done = self.is_done(shot)
    #     score = 0.0
    #     if done:
    #         distance_score = self.get_score()
    #         time_score = - (self.game_steps/(abs(distance_score)+1e5)) # The higher the pace, the slowest it goes
    #         score = distance_score + time_score
    #         self.soft_reload()
    #     return shot.astype(np.float).ravel(), score, done

    def step(self, action):
        self.agent.start_game()
        self.game_steps += 1
        #self.agent.unpause()
        # for char in action:
        #     getattr(self.agent, char)()
        self.agent.step(action)
        shot = self.get_screen_shot()
        #self.agent.pause()
        done = self.is_done(shot)
        distance_score = self.get_score() or 0
        time_score = - (self.game_steps/(abs(distance_score)+1e5)) # The higher the pace, the slowest it goes
        score = distance_score -  self.old_score
        self.old_score = distance_score
        if done:
            self.soft_reload()
            score = -100
        return shot.astype(np.float).ravel(), score, done

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
            """
            TODO:
            TEST ONLY GRAYSCALE
            this processing might not be useful since the important data is in the difference between frames

            img = np.array(shot)
            img[:, :, 2] = 0
            img[:, :, 1] = 0
            blueidx = img[:, :, 0] < 24
            notblueidx = img[:, :, 0] >= 24
            img[blueidx] = 255
            img[notblueidx] = 0
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
             """
            img = np.array(shot)[:,:,0]
            img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            # Grab the data
            # Save to the picture file
            # mss.tools.to_png(shot.rgb, shot.size, output='12345.png')
            if render: self.render(img)
        return img

    def render(self, img):
        cv2.waitKey(1)
