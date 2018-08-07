import numpy as np
import pandas as pd
import data_file
import copy
import Draw
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification
from collections import Counter

action_max, n_actions = 1, 10
reward = np.linspace(0, action_max, n_actions, endpoint=False)
sameCnt = 30
colsadd = list(map(lambda x:"ft"+str(x),np.arange(0,sameCnt,1)))
colsadd.extend(["ftMean","ftMedian","ftStd","ftVar"])

def getBaseAction(t_r):
    d = np.arange(0, n_actions, 1)
    d_ = []
    for i in d:
        if (t_r - reward[i]) >= 0:
            d_.append(i)
    imax = max(d_)
    action = int(imax)
    return action

def reLabel(inPath,outPath,is_double=False):
    ##读入保险人文件
    mycol = copy.deepcopy(data_file.henan_no_city)
    mycol.extend(['Exam_dangr_obj', 'Outcome_normalpre',"ti_real","ri"])
    mycol.extend(colsadd)
    data_train = pd.read_csv(inPath, encoding='gbk',
                           usecols=mycol)
    list_case = []
    for indexs in data_train.index:
        t = data_train.loc[indexs,"ti_real"]
        list_case.append(getBaseAction(t))
    data_train["ti_case"] = pd.Series(list_case)
    #Draw.plotXY([data_train["ti_real"]], [data_train["ti_case"]], "t", "case", "label")

    X_train = data_train[np.array(data_train.columns)[: -1]]
    Y_train = data_train[np.array(data_train.columns)[-1]]
    print(X_train.shape)
    print(Counter(Y_train))

    X_resampled, Y_resampled = RandomOverSampler(random_state=0).fit_sample(X_train, Y_train)
    print(X_resampled.shape)
    print(sorted(Counter(Y_resampled).items()))

    middle = np.column_stack((X_resampled, Y_resampled))
    mycol.extend(["ti_case"])
    dfall = pd.DataFrame(data=middle,columns=mycol)

    if is_double:
        X_train = dfall[np.array(dfall.columns)]
        Y_train = dfall[np.array(dfall.columns)[-4]]
        print(X_train.shape)
        print(sorted(Counter(Y_train).items()))

        X_resampled, Y_resampled = RandomOverSampler(random_state=0).fit_sample(X_train, Y_train)
        print(X_resampled.shape)
        print(sorted(Counter(Y_resampled).items()))

        #middle = np.column_stack((X_resampled, Y_resampled))
        #mycol.extend(["ti_case"])
        dfall = pd.DataFrame(data=X_resampled, columns=mycol)

    dfall =dfall.sample(n=1000)
    dfall.to_csv(outPath, index=False, sep=',')

    X_train = dfall[np.array(dfall.columns)[: -1]]
    Y_train = dfall[np.array(dfall.columns)[-1]]
    print(X_train.shape)
    print(sorted(Counter(Y_train).items()))

def obtainSameLeafT(num,leaf_train,leaf_test,ti_real_list,sameCnt = 30):
    # num:训练集的当前索引
    # 训练叶子的每个元素剪掉测试集中该采样点的叶子值，统计为0，即相同的个数
    tmp = leaf_train - leaf_test[num]
    # 对训练集中每个元素看它总共与该测试数据相同的叶子数目（由不同的决策树产生）
    result_raw = np.sum((tmp == 0) * 1, axis=1)
    result = np.argsort(result_raw)
    # 取倒数30个，即相同最多的，最相似的30个
    result = result[-sameCnt:]
    if result[0] < 10:
        print(result[0], 'ft lowwwwwwwwwwwwwwwwwwwwwwwwwwwwww')
    # result对应的ti_real的list
    ft = ti_real_list[result]
    return ft

def addLeafCols(inPath,outPath,leafPath,trainPath,is_Train=False,in_cnt = 20000): #'new_leaf__v8.npy'
    phi = 0
    ##读入保险人文件
    mycol = copy.deepcopy(data_file.henan_no_city)
    mycol.extend(['Exam_dangr_obj', 'Outcome_normalpre',"ti_real","ri"])
    data_in = pd.read_csv(inPath, encoding='gbk',
                           usecols=mycol) ##train(154039,130) test(38510,130)
    (leaf_train, leaf_test) = np.load(leafPath)
    if not is_Train:
        data_train = pd.read_csv(trainPath, encoding='gbk',
                          usecols=mycol)
    else:
        data_train = data_in

    ti_real_list = np.array((1 - phi) * data_train['ti_real'] + phi * data_train['ri'])

    listadd = []
    print(colsadd)
    if in_cnt > 0:
        data_in = data_in[:in_cnt]
    # 这里增加了筛选
    data_in = data_in[data_in.ti_real<0.3]  ##train(150956,130) test(37778,130)
    data_in = data_in[data_in.ri<0.2]   ##train(150200,130) test(37585,130)


    for i in range(len(data_in)):
        _list = []
        ft = obtainSameLeafT(i,leaf_train,leaf_test,ti_real_list,sameCnt = sameCnt)
        # Draw.drawHist(ft,"ft", ranges=(0, 1))
        ft_arr = np.array(ft)
        _list.extend(ft)
        _list.extend([ft_arr.mean(),np.median(ft_arr),ft_arr.std(),ft_arr.var()])
        listadd.append(_list)
        print(str(i)+str(_list))

    print(data_in.shape)
    dfadd = pd.DataFrame(listadd,columns=colsadd)
    data_in = data_in.join(dfadd)
    print(data_in.shape)

    data_in.to_csv(outPath, index=False, sep=',')


if __name__ == "__main__":
    import os
    os.chdir("data")
    # reLabel('new_train__v8e.csv','new_train__v9e.csv')
    # reLabel('new_test__v8e.csv','new_test__v9e.csv')
    # reLabel('new_train__v8.csv', 'new_train__v10.csv',is_double=True)
    # reLabel('new_test__v8.csv', 'new_test__v10.csv',is_double=True)

    # addLeafCols('new_test__v8.csv', 'new_test__v8f.csv', 'new_leaf__v8.npy',
    #             'new_train__v8.csv', is_Train=False, in_cnt=-2)
    addLeafCols('new_train__v8.csv', 'new_train__v8f.csv', 'new_leaf__v8.npy',
                'new_train__v8.csv', is_Train=False, in_cnt=-2)






