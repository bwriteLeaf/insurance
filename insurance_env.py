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

isSelect = True
#cols_state = ["ti_real"]
cols_state = ["ri"]
ti_select = 0.3
ri_select = 0.2
alpha, mu = 1,1

class Insurance():
    def __init__(self,num_episode = 10000,type = "adjust"):
        self.num_episode = num_episode
        self.n_actions = (10 if type != "adjust" else 3)
        self.n_features = 1+1
        self.action_max = 1
        self.episode_cnt = 0
        self.td_cnt = 0
        self.type = type

        ##读入保险人文件，训练or测试
        #mycol = copy.deepcopy(data_file.henan_no_city)
        #mycol.extend(['Exam_dangr_obj', 'Outcome_normalpre',"ti_real","ri"])
        # 准备需要读取的列名称
        mycol = ['Outcome_normalpre',"ti_real","ri"]
        if type == "action":
            mycol.extend(data_process.colsadd)
        # print(data_process.colsadd)
        # print(self.n_features)
        self.data_raw_train = pd.read_csv('data/new_train__v8.csv', encoding='gbk',
                               usecols=mycol)
        self.data_raw_test = pd.read_csv('data/new_test__v8.csv', encoding='gbk',
                                    usecols=mycol)

        # 进行训练以及测试数据的筛选
        if isSelect:
            self.data_raw_train = self.data_raw_train[self.data_raw_train.ti_real <= ti_select]
            self.data_raw_test = self.data_raw_test[self.data_raw_test.ti_real <= ti_select]
            self.data_raw_train = self.data_raw_train[self.data_raw_train.ri <= ri_select]
            self.data_raw_test = self.data_raw_test[self.data_raw_test.ri <= ri_select]
            self.data_raw_train = self.data_raw_train.reset_index(drop=True)
            self.data_raw_test = self.data_raw_test.reset_index(drop=True)
            self.action_max = ti_select

        ##生成actions
        if type == "action":
            d = list(np.linspace(0,self.action_max,self.n_actions, endpoint=False)) #action的保费值
            e = list(np.full(self.n_actions, 1))#action的赔款值
            e[0] = 0
            self.action_space = list(map(lambda x: (x[0],x[1]), zip(d,e)))
            # print(self.action_space)
        elif type == "adjust":
            self.action_space = [(-0.01,1),(0,1),(0.01,1)]  #注意这里action_space的定义与直接输出动作值是有区别的

        ##定义价格转移的存储结构
        lengthn = num_episode if (num_episode > 0) else len(self.data_raw_train)
        lengtht = num_episode if (num_episode > 0) else len(self.data_raw_test)
        # self.train_price = np.random.rand(lengthn)
        # self.test_price = np.random.rand(lengtht)
        self.train_price = np.random.uniform(0,ti_select,size=lengthn)
        self.test_price = np.random.uniform(0,ti_select,size=lengtht)

        self.data_all = (self.data_raw_train[:num_episode] if (num_episode > 0) else self.data_raw_train)
        self.price_all = self.train_price

        self.stepcnt = 0
        self.stepList = []

    def getState(self,discount=False):
        retlist = list(map(lambda x: self.data_all.loc[(self.choiceNow if not discount else self.choiceNow-1),x], cols_state))

        if self.type == "adjust":
            retlist.append(self.price_all[self.choiceNow])
        else:
        ##增加额外信息
            ft = self.getSameLeafT()
            retlist.append(ft[-4])

        return np.array(retlist)

    def reset(self,Train):
        # return observation from the beginning
        if not Train:
            self.data_all = (self.data_raw_test[:self.num_episode] if (self.num_episode > 0) else self.data_raw_test)
            if self.episode_cnt == 0:
                self.price_all = self.test_price

        # self.stepList = np.random.permutation(len(self.data_all))
        self.stepList = np.arange(0,len(self.data_all),1)
        self.stepcnt = 0
        self.choiceNow = self.stepList[self.stepcnt]
        self.episode_cnt += 1
        beginning = self.getState()
        return beginning

    def step(self, action,isFloat = False):
        s = self.getState()
        # reward function
        if self.type == "adjust":
            priceNow = self.getAction_Income(action,isAcc=True)
            reward, income, pay, n, m, b = self.getAction_Profit(action,isAcc=True)
            self.price_all[self.choiceNow] = priceNow
        else:
            reward,income,pay,n,m,b = self.getAction_Profit(action)
            priceNow = income
        # print("%s t:%f action: %f reward: %f priceNow: %f" %(str(s),self.getT(),action, reward,priceNow) )

        done = (self.stepcnt == len(self.data_all) - 1)
        if not done:
            self.stepcnt += 1
            self.choiceNow = self.stepList[self.stepcnt]
        s_ = self.getState()

        return s_, reward, done


    def getT(self):
        return float(self.data_all.loc[self.choiceNow,"ti_real"])

    def getY(self):
        return float(self.data_all.loc[self.choiceNow,"Outcome_normalpre"])

    def getR(self):
        return float(self.data_all.loc[self.choiceNow,"ri"])

    # 下面函数是可以接受浮点数值的,并且可以接受累计收益的情况
    def getAction_Income(self, action, isFloat=False,isAcc=False):  ## Income  将输入的动作序号转化为动作的赔款额
        if isAcc:
            a = self.price_all[self.choiceNow]+self.action_space[action][0]
            a = max(a,0)
            return float(a)
        if not isFloat:
            a = self.action_space[action][0]
        else:
            a = action
        return float(a)

    def getAction_Payment(self, action, isFloat=False,isAcc=False):  ##must be before the next step
        y = self.getY()
        payment = float(0 if self.getAction_Income(action,isFloat,isAcc) == 0 else y)

        return payment

    def getAction_Profit(self, action, isFloat=False,isAcc=False):  ##
        t = self.getT()
        r = self.getR()
        get = self.getAction_Income(action, isFloat,isAcc)
        pay = self.getAction_Payment(action, isFloat,isAcc)
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
        return float(profit_cmp), float(income_cmp), float(pay_true_cmp), float(value_cmp), float(profit_p), float(
            value_p)
    def getVolume(self,profit): ## Income 判断收益是否为0，如果为0说明交易没有达成
        return (1 if profit != 0 else 0)

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

    def getBaseAction_fstage(self): ##这个版本是限制在离、离散取值的
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

    def getBaseAction_f(self):
        t_r = self.getT()
        e_t = self.getState()[0]

        return t_r
        # return e_t

    def getSameLeafT(self):
        raw = self.data_all.loc[self.choiceNow, data_process.colsadd]
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









