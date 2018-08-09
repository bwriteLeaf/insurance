"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

#import gym
from 强化学习代码.RL_model_v3 import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from 强化学习代码.myenv_v3 import MyEnv

np.random.seed(1)
DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

ss = '__v8'  # '_500_300_v3'
data_train = pd.read_csv('/Users/apple/Documents/研究生/insurance/insurance/data/new_train' + ss + '.csv', encoding='gbk',usecols=['ri','Outcome_normalpre','ti_real'])
print(data_train.shape)
# 从一个均匀分布[low,high)中随机采样
data_train['zi']=np.random.uniform(0,0.3,size=len(data_train))#data_train['ri']
data_train=data_train[['ri','zi','Outcome_normalpre','ti_real']]

#action_list=[0,0.1,0.2,0.3]

env = MyEnv(data_train)

RL = PolicyGradient(
    n_actions=env.action_space,
    n_features=env.observation_space,
    learning_rate=0.001,
    reward_decay=0.85,
    # output_graph=True,
)

for i_episode in range(60):

    # 每隔3轮进行一次表现的输出
    if i_episode%3==0:
        # TODO：
        prob = RL.prob_result(np.array(env.data)[:, :-2])
        actions = np.argmax(prob, axis=1)
        #
        #np.save('mypg_result.npy', env.data)
        data_ti=env.data[:,-1]
        data_zi=env.data[:,-3]
        data_label=env.data[:,-2]
        num_sum = np.sum((data_ti>data_zi))
        sums = np.sum((data_zi - data_label) * (data_ti >= data_zi))

        print(i_episode,'actions:',sum(actions == 0), sum(actions == 1), sum(actions == 2),num_sum, sums)

    observation = env.reset()

    # 每一集里步长的计数
    ii=1
    while True:
        #if RENDER: env.render()
        if ii%50000==0:
            pass
            # print(ii)
        ii+=1
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            '''
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            #if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            #print("episode:", i_episode, "  reward:", int(running_reward))
            '''
            vt = RL.learn()

            break

        observation = observation_



data_test = pd.read_csv('/Users/apple/Documents//研究生/insurance/insurance/data/new_test' + ss + '.csv', encoding='gbk',usecols=['ri','Outcome_normalpre','ti_real'])
data_test['zi']=np.random.uniform(0,0.3,size=len(data_test))#data_test['ri']
data_test=data_test[['ri','zi','Outcome_normalpre','ti_real']]
env = MyEnv(data_test)
for i_episode in range(60):

    if i_episode%3==0:
        prob = RL.prob_result(env.data[:, :-2])
        actions = np.argmax(prob, axis=1)

        np.save('mypg_result.npy', env.data)

        data_ti = env.data[:, -1]
        data_zi = env.data[:, -3]
        data_label = env.data[:, -2]
        num_sum = np.sum((data_ti > data_zi))
        sums = np.sum((data_zi - data_label) * (data_ti >= data_zi))

        print(i_episode,'actions:',sum(actions == 0), sum(actions == 1), sum(actions == 2),num_sum, sums)

    observation = env.reset_test()

    ii=1
    while True:
        #if RENDER: env.render()
        if ii%50000==0:
            pass
            # print(ii)
        ii+=1
        action = RL.choose_action_test(observation)
        observation_,done = env.step_test(action)
        if done:
            break

        observation = observation_
#data_test = pd.read_csv('new_test' + ss + '.csv', encoding='gbk')
prob=RL.prob_result(np.array(data_test)[:,:-2])
actions=np.argmax(prob,axis=1)

np.save('mypg_result.npy', env.data)
data_ti=env.data[:,-1]
data_zi=env.data[:,-3]
data_label=env.data[:,-2]
num_sum = np.sum((data_ti>data_zi))
sums = np.sum((data_zi - data_label) * (data_ti >= data_zi))

print(num_sum, sums)