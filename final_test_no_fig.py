# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
#import xgboost as xgb
import datetime
# import pre_train_test
import csv
import data_file
import copy
import multiprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from scipy.stats import beta
import sys


def GussianParzen(f, x, h):
    N = len(x)
    f = sorted(f)
    b = 0.0
    h1 = h / np.sqrt(N)
    pf = []
    cf = []
    cf_sum = 0
    for i in range(0, len(f)):
        b = np.exp(((x - f[i]) / h1) * ((x - f[i]) / h1) / (-2.0)) / np.sqrt(2.0 * np.pi) / h1
        b = np.sum(b)
        '''
        for j in range(0,N):
            #rint (x[j]-f[i])/h1
            b= b+ math.exp(((x[j]-f[i])/h1)*((x[j]-f[i])/h1)/(-2.0))/math.sqrt(2.0*math.pi)/h1
        '''
        pf.append(b)
        cf_sum += b
        cf.append(cf_sum)
    pf = pf / cf[-1] / (f[1] - f[0])
    cf = cf / cf[-1]
    return pf, cf


def get_z(t, c):
    # 不存在c大于0的，直接返回无结果
    if len(np.where(c > 0)[0]) == 0:
        print('no zi-----')
        return None
    # 不存在c小于0的，直接取临界值即第一个满足条件的下界值
    if len(np.where(c < 0)[0]) == 0:
        return t[0]
    # 大于0中最小的c索引值（也是t索引值）和小于0中最大的
    min_1 = np.min(np.where(c > 0)[0])
    max_0 = np.max(np.where(c < 0)[0])
    # 两者相减只要大于－0.1即可，按常理而言c是单调的该值应该大于0，不会出现warning
    if t[min_1] >= t[max_0] - 0.1:
        return max(t[min_1], t[max_0])
    elif t[min_1] >= t[max_0] - 0.25:
        print(t[min_1], t[max_0], 'warning---------------------------------------------')
        # np.save('war_'+str(num)+'.npy',(t,c))
        return max(t[min_1], t[max_0])
    else:
        print(t[min_1], t[max_0], 'err，no zi-------------------------------------------')
        # np.save('err_'+str(num)+'.npy',(t,c))

    return None


def get_c(pf, cf, t, ri, alpha, mu):
    # mu * (t - ri) - (mu - 1 + mu) * ((1 - cf)) / pf
    return mu * (alpha * t - ri) - (mu * alpha - 1 + mu) * ((1 - cf)) / pf  # -(alpha)*((1-cf))+(alpha*t-ri)*pf


import os


def removeFileInDir(targetDir):
    if not os.path.exists(targetDir):
        os.mkdir(targetDir)
    '''
    for file in os.listdir(targetDir):
        targetFile = os.path.join(targetDir, file)
        if os.path.isfile(targetFile):
            print('Delete Old Log FIle:', targetFile)
            os.remove(targetFile)
    '''



def model_run(params):
    data_train, data_test, leaf_train, leaf_test, dir_name, alpha, mu, phi,deta=params
    # 目前设置除了phi为0，其它三个参数都是1
    count = 0
    # 定价list，长度为测试集数据长度
    zi_list = np.zeros((data_test.shape[0],))
    count_i = 0
    step_count =5000#这个是测试集的样本数量
    print()
    # 按一定比例加入范围在－1到1之间的随机噪声，加入程度由(1-deta)决定 从一个均匀分布[low,high)中随机采样，size: 输出样本数目
    ri_list = deta*np.array(data_test['ri'])+(1-deta)*np.random.uniform(-1,1,len(data_test))
    ri_list[ri_list>1]=1
    ri_list[ri_list<0]=0
    # 按一定比例在'ti_real'中掺杂真实的'ri'，掺杂程度由phi决定
    ti_real_list= np.array((1-phi)*data_train['ti_real']+phi*data_train['ri'])
    # 例如，data长度为100，step为5，则迭代步长为20，迭代［0，20，40，60，80］共5个元素
    # 即一共是5000个元素，每2－3步采一个点
    for num in range(0, len(data_test), int(len(data_test) / step_count)):  # sample_list:
        ri = ri_list[num]
        # 训练叶子的每个元素剪掉测试集中该采样点的叶子值，统计为0，即相同的个数
        tmp = leaf_train - leaf_test[num]
        # 对训练集中每个元素看它总共与该测试数据相同的叶子数目（由不同的决策树产生）
        # 有改动！！！
        result_raw = np.sum((tmp == 0) * 1, axis=1)
        result = np.argsort(result_raw)
        # 取倒数30个，即相同最多的，最相似的30个
        result = result[-30:]

        # print(result[-10:])
        if result[0] < 10:
            print(result[0], 'ft lowwwwwwwwwwwwwwwwwwwwwwwwwwwwww')

        # result对应的ti_real的list
        ft = ti_real_list[result]
        # 0.99以上规范到1，0.02以下规范到0.01
        # 有改动！！！0.01以下规范到0.01
        ft_min = max(0.01, np.min(ft))
        ft_max = min(1, np.max(ft) + 0.01)
        ranges = ft_max - ft_min
        # ti_real在范围以内，步长为0.0001
        t = np.arange(ft_min, ft_max, 0.0001)
        # pf, cf与t的长度一致；pf在0以上，cf在1以内且升序排列
        pf, cf = GussianParzen(t, ft, ranges)
        # 与t的长度一致，从正数到负数都有
        c_set = get_c(pf, cf, t, ri, alpha, mu)
        zi = get_z(t, c_set)

        if zi != None:
            zi_list[num] = zi
        else:
            # 2应该是一个不可能达到要求的价格
            zi_list[num] = 2
        if count_i%500==0:
            print(count_i)
        count_i += 1

    print('final:',count_i)

    # 所有定价大于0的元素索引
    index_list=np.where(zi_list>0)[0]
    # 所有定价大于0的元素的妊娠结局
    norm_label_new = np.array(data_test['Outcome_normalpre'])[index_list]
    ri_new=ri_list[index_list]
    ti_real_new = ti_real_list[index_list]
    zi_new=zi_list[index_list]
    # 所有成交了的元素索引
    deal_index=np.where(zi_new<=ti_real_new)[0]
    # 总效用，mu代表公司效用的占比
    value_sum = np.sum((mu * (alpha * zi_new - ri_new) + (1 - mu) * (ti_real_new - zi_new))[deal_index])
    # 被保人效用
    value_ti = np.sum((ti_real_new - zi_new)[deal_index])
    # 被保人实际收益
    income_ti = np.sum((norm_label_new - zi_new)[deal_index])

    # 公司效用
    value_ri = np.sum((alpha * zi_new - ri_new)[deal_index])
    # 公司实际收益
    income_ri = np.sum((alpha * zi_new - norm_label_new)[deal_index])

    print('mu:',mu)
    print('numbers:', len(deal_index))
    print('value_ti:', value_ti)
    print('income_ti,', income_ti)
    print('value_ri:', value_ri)
    print('income_ri,', income_ri)
    print('value:', value_sum)
    print('claim:', np.sum(norm_label_new[deal_index]))


    np.save(dir_name + 'pay_income_alpha'+str(alpha)+'_mu'+str(mu)+'_phi'+str(phi)+'_deta'+str(deta)+'.npy',
            (norm_label_new, zi_new, ti_real_new,ri_new))

    log_file = open(dir_name + 'reult_alpha'+str(alpha)+'_mu'+str(mu)+'_phi'+str(phi)+'_deta'+str(deta)+'.txt', "w")

    #print("Now all print info will be written to message.log")
    log_file.write(str(len(index_list)))
    log_file.write('\nnumbers:'+str(len(deal_index)))
    log_file.write('\nvalue_ti:' + str(value_ti) )
    log_file.write('\nincome_ti,' + str(income_ti))
    log_file.write('\nvalue_ri:' + str(value_ri) )
    log_file.write('\nincome_ri,' + str(income_ri))
    log_file.write('\nvalue:'+str(value_sum))
    log_file.write('\nclaim:'+str(np.sum(norm_label_new[deal_index])))
    log_file.close()


if __name__ == "__main__":

    #以下四个参数不需要改
    alpha = 1#这个参数是保险公司通过保费获得收益的参数，论文中用omega，为1表示没有收益
    mu = 1
    deta=1
    phi = 0
    ss = '__v8'  # '_500_300_v3'
    data_train = pd.read_csv('new_train' + ss + '.csv', encoding='gbk')
    data_test = pd.read_csv('new_test' + ss + '.csv', encoding='gbk')
    # w_col = data_file.henan_no_city

    # model_t_r = joblib.load('new_model_r'+ss+'.m')
    # model_t = joblib.load('new_model_t'+ss+'.m')

    # 随即森林预测妊娠结局的叶子结果
    (leaf_train, leaf_test) = np.load('new_leaf' + ss + '.npy')

    dir_name = '.\\fig_deta_new\\'
    removeFileInDir(dir_name)


    params = []
    #for detas in range(0,11):
    #   deta=detas/10
    params.append([data_train, data_test, leaf_train, leaf_test, dir_name, alpha, mu ,phi,deta])#[原始数据文件夹路径，输出文件路径]

    starttime = datetime.datetime.now()
    #多线程处理数据，加入参数运行
    pool = multiprocessing.Pool(processes=4)
    pool.map(model_run, params)
    pool.close()
    pool.join()

    print('Pooling over!......')

    print(data_test.shape)
    #model_run(data_train, data_test, leaf_train, leaf_test, dir_name, alpha, mu ,phi)
    endtime = datetime.datetime.now()

    print(endtime - starttime)
