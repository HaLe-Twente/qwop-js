from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import numpy as np
import time
from pynput.keyboard import Key, Controller

class Agent2:

    def __init__(self):
        chrome_opts = Options()

        # REFER TO https://github.com/zalando/zalenium/issues/497 for enabling flash in docker
        chrome_opts.add_argument('--disable-features=EnableEphemeralFlashPermission')
        chrome_opts.add_argument('--disable-infobars')
        chrome_opts.add_argument("--ppapi-flash-version=32.0.0.101")
        chrome_opts.add_argument("--ppapi-flash-path=/usr/lib/pepperflashplugin-nonfree/libpepflashplayer.so")


        chrome_prefs = {"profile.default_content_setting_values.plugins": 1,
                        "profile.content_settings.plugin_whitelist.adobe-flash-player": 1,
                        "profile.content_settings.exceptions.plugins.*,*.per_resource.adobe-flash-player": 1,
                        "PluginsAllowedForUrls": "localhost"}
        chrome_opts.add_experimental_option('prefs', chrome_prefs)
        self.driver = webdriver.Chrome(executable_path = './chrome-driver/chromedriver', chrome_options = chrome_opts)
        self.driver.get('localhost:3000')
        self.canvas = self.driver.find_element_by_id('window1')
        self.canvas_size = {"w":640, "h":400}
        self.is_paused = False
        self.last_action=[-1,-1,-1,-1]
        self.is_pressed = [False, False, False, False]
        self.keyboard = Controller()

    def step(self, action):
        #print(action)
        action = np.sign(action)
        action = [-1 if key < 0 else 1 for key in action]
        i = 0
        for new, old in zip(action, self.last_action):
            if new != old:
                if i == 0:
                    self.q()
                elif i == 1:
                    self.w()
                elif i == 2:
                    self.o()
                elif i == 3:
                    self.p()
            i += 1

    def q(self):
        if self.is_pressed[0]:
            self.keyboard.release('q')
            self.is_pressed[0] = False
        else:
            self.keyboard.press('q')
            self.is_pressed[0] = True

    def w(self):
        if self.is_pressed[1]:
            self.keyboard.release('w')
            self.is_pressed[1] = False
        else:
            self.keyboard.press('w')
            self.is_pressed[1] = True

    def o(self):
        if self.is_pressed[2]:
            self.keyboard.release('o')
            self.is_pressed[2] = False
        else:
            self.keyboard.press('o')
            self.is_pressed[2] = True

    def p(self):
        if self.is_pressed[3]:
            self.keyboard.release('p')
            self.is_pressed[3] = False
        else:
            self.keyboard.press('p')
            self.is_pressed[3] = True


    def r(self):
        choice = np.random.choice(['q','w','o','p'])
        self.keyboard.press(choice)
        time.sleep(0.08)
        self.keyboard.release(choice)

    def space(self):
        self.keyboard.press(Key.space)
        time.sleep(0.08)
        self.keyboard.release(Key.space)

    def start_game(self):
        self.canvas = self.driver.find_element_by_id('window1')
        self.canvas.click()

    def click_tutorial(self):
        ac = ActionChains(self.driver)
        ac.move_to_element(self.canvas)
        ac.move_by_offset(-self.canvas_size["w"]/2+10, -self.canvas_size["h"]/2+20)
        ac.click()
        ac.perform()

    def pause(self):
        self.click_tutorial()

    def unpause(self):
        self.click_tutorial()

    def hard_reload(self):
        self.keyboard.press(Key.f5)
        time.sleep(0.08)
        self.keyboard.release(Key.f5)

    def reload(self):
        self.keyboard.press('r')
        time.sleep(0.08)
        self.keyboard.release('r')

    def screen_shot(self):
        return self.canvas.screenshot_as_base64

    def game_ended(self):
        return self.driver.execute_script("return window.gameEnded;")

    def get_score(self):
        score = self.driver.execute_script("return window.score;")
        if score is None:
            score = 0
        return score

