from insurance_env import Insurance
from RL_DQN import DeepQNetwork
import Draw
import numpy as np
import pandas as pd
from DDPG_update2 import DDPG
#from RL_brain import DuelingDQN
#from DQN_modified import DeepQNetwork
import os

replace_target_iter=200
episode_cnt = 20
episode_cnt_test = 2
insurance = True
LEGEND = ["agent", "baseline", "baseline_f","theory","random"]
n = 4
is_Theory = True
MEMORY_CAPACITY = 10000
ss = "DDPG"

def runTest(Train = True):
    step = 0  ##跨越episode的总步数计数
    t_all = [] ##agent,baseline,baseline_f,random,random2
    r_all = []
    x_all = [[],[],[],[],[]]
    profit_all = [[],[],[],[],[]]
    pay_all = [[],[],[],[],[]]

    volume_episode = [[], [], [], [],[]]
    profit_episode = [[], [], [], [], []]
    pay_episode = [[], [], [], [], []]
    income_episode = [[], [], [], [], []]
    profit_p_episode = [[], [], [], [], []]
    value_episode = [[], [], [], [], []]
    value_p_episode = [[], [], [], [], []]

    for episode in range(episode_cnt_test):
        # initial observation
        observation = env.reset(test=True)  ##feature数组
        volume_episode_cnt = [0, 0, 0, 0,0]
        pay_episode_cnt = [0, 0, 0, 0, 0]
        income_episode_cnt = [0, 0, 0, 0, 0]
        profit_episode_cnt = [0, 0, 0, 0, 0]
        profit_p_episode_cnt = [0, 0, 0, 0, 0]
        value_episode_cnt = [0, 0, 0, 0, 0]
        value_p_episode_cnt = [0, 0, 0, 0, 0]
        var = 3  # control exploration
        actions = []

        while True:
            # RL choose action based on observation
            if insurance:
                arr = ddpg.choose_action(observation)
                arr = np.clip(np.random.normal(arr, var), -1,1)
                action_value = np.clip(int((float(arr[0])+1)/2*10)/10,0,0.9)
                action = arr
            else:
                action = ddpg.choose_action(observation)
                action = np.clip(np.random.normal(action, var), 0,1)

            r = observation[0]
            r_all.append(r)
            t_all.append(env.getT())
            x_all[0].append(env.getAction_Income(action_value, isFloat=True))  ##x 仅仅是定价 ，后面的profit才是真是收益
            x_all[1].append(env.getAction_Income(env.getBaseAction()))
            x_all[2].append(env.getAction_Income(env.getBaseAction_f()))


            pf_cmp0,i_cmp0,py_cmp0,v_cmp0,pf_p0,v_p0 = env.getAction_Profit(action_value, isFloat=True)
            pf_cmp1,i_cmp1,py_cmp1,v_cmp1,pf_p1,v_p1 = env.getAction_Profit(env.getBaseAction())
            pf_cmp2, i_cmp2, py_cmp2,v_cmp2,pf_p2,v_p2 = env.getAction_Profit(env.getBaseAction_f())
            #
            if is_Theory:
                pf_cmp3, i_cmp3, py_cmp3,v_cmp3,pf_p3,v_p3 = env.getAction_Profit(env.getTheoryPrice(),isFloat=True)
                x_all[3].append(env.getAction_Income(env.getTheoryPrice(), isFloat=True))
            else:
                pf_cmp3, i_cmp3, py_cmp3, v_cmp3, pf_p3, v_p3 = 0,0,0,0,0,0

            i_cmp_list = [i_cmp0,i_cmp1,i_cmp2,i_cmp3]
            py_cmp_list = [py_cmp0, py_cmp1, py_cmp2, py_cmp3]
            pf_cmp_list = [pf_cmp0, pf_cmp1, pf_cmp2, pf_cmp3]
            v_cmp_list = [v_cmp0, v_cmp1, v_cmp2, v_cmp3]
            pf_p_list = [pf_p0, pf_p1, pf_p2, pf_p3]
            v_p_list = [v_p0, v_p1, v_p2, v_p3]
            volume_list = list(map(lambda x: env.getVolume(x), pf_cmp_list))

            income_episode_cnt = addToList(income_episode_cnt,i_cmp_list,n)
            pay_episode_cnt = addToList(pay_episode_cnt, py_cmp_list, n)
            profit_episode_cnt = addToList(profit_episode_cnt,pf_cmp_list , n)
            volume_episode_cnt = addToList(volume_episode_cnt,volume_list, n)
            profit_p_episode_cnt = addToList(profit_p_episode_cnt, pf_p_list, n)
            value_episode_cnt = addToList(value_episode_cnt, v_cmp_list, n)
            value_p_episode_cnt = addToList(value_p_episode_cnt, v_p_list, n)

            pay_all = appendToList(pay_all, [py_cmp0, py_cmp1, py_cmp2, py_cmp3], n)
            profit_all = appendToList(profit_all, [pf_cmp0, pf_cmp1, pf_cmp2, pf_cmp3], n)

            # RL take action and get next observation and reward
            ##reward int , done boolean
            observation_, r, done = env.step(action_value,isFloat=True)
            if Train:
                ddpg.store_transition(observation, action, r / 10, observation_)

                # 总共学了200步以上后，每隔5步进行一次学习
                if ddpg.pointer > MEMORY_CAPACITY:
                    var *= .9995  # decay the action randomness
                    ddpg.learn()

            observation = observation_


            # break while loop when end of this episode
            if done:
                # if (episode == episode_cnt-1):
                    # Draw.drawHist(actions, "actions", ranges=(0, 16))
                for i in range(len(LEGEND)):
                    profit_episode[i].append(profit_episode_cnt[i])
                    volume_episode[i].append(volume_episode_cnt[i])
                    pay_episode[i].append(pay_episode_cnt[i])
                    income_episode[i].append(income_episode_cnt[i])
                    profit_p_episode[i].append(profit_p_episode_cnt[i])
                    value_episode[i].append(value_episode_cnt[i])
                    value_p_episode[i].append(value_p_episode_cnt[i])

                break
            step += 1

    # end of game
    print('game over')
    text = ("" if not Train else "_train")
    drawXT([r_all], t_all, LEGEND, "r", "t"+text, avg=False)
    Draw.plotMulti(profit_episode[:n], "rewards_episode"+text, legend=LEGEND[:n])
    if not Train:
        drawXT(x_all, r_all, LEGEND, "r", "x", avg=True)
        drawXT(profit_all, r_all, LEGEND, "r", "profit", avg=True)
        drawXT(x_all, t_all, LEGEND, "t", "x", avg=True)
        drawXT(profit_all, t_all, LEGEND, "t", "profit", avg=True)

        Draw.plotMulti(volume_episode[:n], "volume_episode_test", legend=LEGEND[:n])
        Draw.plotMulti(pay_episode[:n], "pay_episode_test", legend=LEGEND[:n])
        Draw.plotMulti(income_episode[:n], "income_episode_test", legend=LEGEND[:n])

    print_list = list(map(lambda i: list(map(lambda x:np.mean(x),[profit_episode[i],pay_episode[i],income_episode[i],
                                     volume_episode[i],value_episode[i],
                                     profit_p_episode[i],value_p_episode[i]])),
                      np.arange(0,len(LEGEND[:n]),1)))
    print_df = pd.DataFrame(print_list,columns=["profit","pay","income",
                                                "volume","value","profit_p","value_p"],
                            index=LEGEND[:n])
    print_df.to_csv("results/print-list"+text+".csv", index=True, sep=',')



def addToList(list,addlist,n):
    for i in range(n):
        list[i] += addlist[i]
    return list

def appendToList(list,appendlist,n):
    for i in range(n):
        list[i].append(appendlist[i])
    return list

def drawXT(x_list,t,legend,xlabel,ylabel,avg=True):
    t_draw_list, x_draw_list = [],[]
    for raw_x in x_list:
        if len(raw_x) !=0:
            if avg:
                t_draw,x_draw = getAvgXT(t,raw_x)
            else:
                t_draw, x_draw = t,raw_x
            t_draw_list.append(t_draw)
            x_draw_list.append(x_draw)
    Draw.plotXY(t_draw_list,x_draw_list,xlabel,ylabel+("-avg" if avg else ""),legend)

def getAvgXT(raw_t,raw_x,):
    ret_x = []
    ret_t = []
    cnt = []
    for i in range(len(raw_t)):
        t = raw_t[i]
        x = raw_x[i]
        if t in ret_t:
            idx = ret_t.index(t)
            ret_x[idx] += x
            cnt[idx] += 1
        else:
            ret_x.append(x)
            ret_t.append(t)
            cnt.append(1)

    for i in range(len(ret_x)):
        ret_x[i] = ret_x[i]/cnt[i]
    return ret_t,ret_x

if __name__ == "__main__":
    # maze game
    path = "results/" + ss
    env = Insurance(num_episode=100)
    s_dim = env.n_features
    a_dim = 1
    a_bound = 1

    ddpg = DDPG(a_dim, s_dim, a_bound)
    if not os.path.exists(path):
        os.mkdir(path)
    Draw.savePath = path
    runTest()
    runTest(Train=False)