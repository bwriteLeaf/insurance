"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

"""
import data_file
import copy
import numpy as np
import pandas as pd
import time
import sys
import Draw
import final_test_no_fig as Theory
import data_process

isSelect = False
#cols_state = ["ti_real"]
cols_state = ["ri"]
ti_select = 0.2
alpha, mu = 1,1

class Insurance():
    def __init__(self,num_episode = 10000):
        self.num_episode = num_episode
        self.n_actions = 10
        self.n_features = 1+1
        self.action_max = 1
        self.episode_cnt = 0
        self.td_cnt = 0

        ##读入保险人文件，训练or测试
        #mycol = copy.deepcopy(data_file.henan_no_city)
        #mycol.extend(['Exam_dangr_obj', 'Outcome_normalpre',"ti_real","ri"])
        mycol = ['Outcome_normalpre',"ti_real","ri"]
        mycol.extend(data_process.colsadd)
        print(data_process.colsadd)
        print(self.n_features)
        self.data_raw_train = pd.read_csv('new_train__v8f.csv', encoding='gbk',
                               usecols=mycol)
        self.data_raw_test = pd.read_csv('new_test__v8f.csv', encoding='gbk',
                                    usecols=mycol)
        if isSelect:
            self.data_raw_train = self.data_raw_train[self.data_raw_train.ti_real <= ti_select]
            self.data_raw_test = self.data_raw_test[self.data_raw_test.ti_real <= ti_select]
            self.action_max = ti_select
        self.data_all = self.data_raw_train[:num_episode]
        self.stepcnt = 0

        ##生成actions
        d = list(np.linspace(0,self.action_max,self.n_actions, endpoint=False))
        #d = np.array([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.125,0.15,0.175,0.2,0.4,0.6,0.8])
        e = list(np.full(self.n_actions, 1))
        e[0] = 0
        self.action_space = list(map(lambda x: (x[0],x[1]), zip(d,e)))
        print(self.action_space)
        #Draw.drawHist(self.data_all["ti_real"],"ti_real",ranges=(0,0.9))

        self.mean = 0.08
    def reset(self,test=False):
        # return observation from the beginning
        if test:
            self.data_all = self.data_raw_test[:self.num_episode]
        else: ##for train
            if self.episode_cnt%200 == 0:  ## 200step change a batch of people
                self.td_cnt += 1
                self.data_all = self.data_raw_train[self.num_episode*(self.td_cnt-1):self.num_episode*self.td_cnt]
                self.data_all = self.data_all.reset_index(drop=True)

        self.stepcnt = 0
        self.episode_cnt += 1
        beginning = self.getState()

        return beginning

    def getState(self,discount=False):
        retlist = list(map(lambda x: self.data_all.loc[(self.stepcnt if not discount else self.stepcnt-1),x], cols_state))

        ##增加额外信息
        ft = self.getSameLeafT()
        # retlist.extend(ft)
        retlist.append(ft[-4])

        return np.array(retlist)

    def getT(self):
        return float(self.data_all.loc[self.stepcnt,"ti_real"])

    def getY(self):
        return float(self.data_all.loc[self.stepcnt,"Outcome_normalpre"])

    def getR(self):
        return float(self.data_all.loc[self.stepcnt,"ri"])

    def getBaseAction(self):
        r = self.getR()
        d = np.arange(0, self.n_actions, 1)
        d_ = []
        for i in d:
            #if (self.action_space[i][1] * r - self.action_space[i][0]) >= 0:
            #    d_.append(i)
            if (self.action_space[i][0] - r) >= 0:
                d_.append(i)
        #imax = max(d_)
        imin = min(d_)
        imax = max(d_)
        ir = np.random.randint(imin, imax + 1)
        action = int(imin)
        return action

    def getBaseAction_f(self):
        r = self.getR()
        t_r = self.getT()
        d = np.arange(0, self.n_actions, 1)
        d_ = []
        for i in d:
            x = self.action_space[i][0]
            if ((x- r) >= 0) and ((1 * t_r - x) >= 0):
                d_.append(i)
        if len(d_) == 0:
            action = 0
        else:
            imax = max(d_)
            action = int(imax)
        return action

    def getSameLeafT(self):
        raw = self.data_all.loc[self.stepcnt, data_process.colsadd]
        ft_raw = list(map(lambda x: raw[x],data_process.colsadd))
        # ft = ft_raw[:data_process.sameCnt]
        return ft_raw

    def getTheoryPrice(self):
        ri = self.getR()
        # result对应的ti_real的list
        ft = self.getSameLeafT()[:data_process.sameCnt]
        # 0.99以上规范到1，0.02以下规范到0.01
        ft_min = max(0.01, np.min(ft))
        ft_max = min(1, np.max(ft) + 0.01)
        ranges = ft_max - ft_min
        # ti_real在范围以内，步长为0.0001
        t = np.arange(ft_min, ft_max, 0.0001)
        # pf, cf与t的长度一致；pf在0以上，cf在1以内且升序排列
        pf, cf = Theory.GussianParzen(t, ft, ranges)
        # 与t的长度一致，从正数到负数都有
        c_set = Theory.get_c(pf, cf, t, ri, alpha, mu)
        zi = Theory.get_z(t, c_set)

        if zi != None:
            return zi
        else:
            # 2应该是一个不可能达到要求的价格
            return 2

    def getRadomAction(self):
        n_max = self.getBaseAction()
        return np.random.randint(0, n_max+1)

    def getVolume(self,profit): ## Income 判断收益是否为0，如果为0说明交易没有达成
        return (1 if profit != 0 else 0)

    # 下面函数是可以接受浮点数值的
    def getAction_Income(self,action,isFloat=False): ## Income  将输入的动作序号转化为动作的赔款额
        if not isFloat:
            a = self.action_space[action][0]
        else:
            a = action
        return float(a)

    def getAction_Payment(self,action,isFloat=False): ##must be before the next step
        y = self.getY()
        if not isFloat:
            payment = float(0 if self.getAction_Income(action) == 0 else y)
        else:
            payment = float(0 if action == 0 else y)
        return payment

    def getAction_Profit(self,action,isFloat=False): ##
        t = self.getT()
        r = self.getR()
        get = float(action if isFloat else self.getAction_Income(action))
        pay = self.getAction_Payment(action,isFloat)
        if ((1 * t - get) < 0):
            profit_cmp = 0
            income_cmp = 0
            pay_true_cmp = 0
            # 保险公司、被保人效用函数
            value_cmp = 0
            profit_p = 0
            value_p = 0
        else:
            profit_cmp = get - pay
            income_cmp = get
            pay_true_cmp = pay
            value_cmp = get - r
            profit_p = pay - get
            value_p = t - get
        return float(profit_cmp),float(income_cmp),float(pay_true_cmp),float(value_cmp),float(profit_p),float(value_p)


    def step(self, action,isFloat = False):
        s = self.getState()
        # reward function
        reward,income,pay,n,m,b = self.getAction_Profit(action,isFloat=isFloat)
        print("%s t:%f action: %f reward: %f base: %f basereward:%f" %(str(s),self.getT(),action, reward,self.getBaseAction(),self.getAction_Profit(self.getBaseAction())[0]) )
        #print("t:%f r:%f Outcome:%f action: %f reward: %f" %(t_r,s[1],y,action, reward) )

        done = (self.stepcnt == len(self.data_all) - 1)

        if not done:
            self.stepcnt += 1
        s_ = self.getState()

        return s_, reward, done





