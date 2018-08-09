import logging
import random
#import gym
import numpy as np
import copy

logger = logging.getLogger(__name__)


class MyEnv:
    def __init__(self,data):


        self.data = np.array(data)
        self.states = None  # 状态空间 -1为ti，-2为label

        self.actions =[0,1,2]#0,下降，1.保持，2，上升
        self.gamma = 0.8  # 折扣因子c
        self.viewer = None
        self.state = None
        # 数据集的大小
        self.num = self.data.shape[0]
        self.index = 0
        self.action_space = len(self.actions)
        self.observation_space = self.data.shape[1] - 2

    def step(self, action):
        # 系统当前状态
        state = self.state
        if self.index == self.num-1:
            return copy.deepcopy(state[:-2]), 1, True, {}

        zi=state[-3]
        if action==2:
            zi+=0.01
        if action==0:
            zi-=0.01
        # 保存价格的时候限制大于等于0！
        self.data[self.index][-3] =max(zi,0)
        ti=state[-1]
        label=state[-2]
        if zi <= ti:
            r = zi - label
        else:
            r = 0
        self.index+=1
        next_state = self.data[self.index]
        self.state = next_state

        is_terminal = False
        if self.index == self.num-1:
            is_terminal = True

        return copy.deepcopy(next_state[:-2]), r, is_terminal, {}

    def reset(self):
        # 数据随机打乱
        np.random.shuffle(self.data)
        self.state = self.data[0]
        self.index = 0
        return copy.deepcopy(self.state[:-2])

    def step_test(self, action):
        # 系统当前状态
        state = self.state
        if self.index == self.num-1:
            return copy.deepcopy(state[:-2]), True

        zi=state[-3]
        if action==2:
            zi+=0.01
        if action==0:
            zi-=0.01
        self.data[self.index][-3] = max(zi,0)
        self.index+=1
        next_state = self.data[self.index]
        self.state = next_state

        is_terminal = False

        if self.index == self.num-1:
            is_terminal = True

        return copy.deepcopy(next_state[:-2]),is_terminal

    def reset_test(self):

        self.state = self.data[0]
        self.index = 0
        return copy.deepcopy(self.state[:-2])