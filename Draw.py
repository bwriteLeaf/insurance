import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

savePath = ""

def plotOne(rewards, ylabel):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(rewards)), rewards)
    ax.set(xlabel = 'training episodes',ylabel = ylabel)
    fig.savefig(os.path.join(savePath,ylabel+"_one.png"))

def plotMulti(re_list,ylabel,legend):
    fig, ax = plt.subplots()
    for rewards in re_list:
        ax.plot(np.arange(len(rewards)), rewards)
    ax.set(xlabel='training episodes', ylabel=ylabel)
    ax.legend(legend)
    fig.savefig(os.path.join(savePath,ylabel+"_multi.png"))

def plotXY(x_list,y_list,xlabel,ylabel,legend):
    fig, ax = plt.subplots()
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        ax.plot(x, y, 'o', ms=1)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(legend)
    #plt.show()
    fig.savefig(os.path.join(savePath,xlabel+"_"+ylabel+".png"))



# 绘制直方图
def drawHist(heights, ylabel, ranges=(0, 200)):
    # 创建直方图
    # 第一个参数为待绘制的定量数据，不同于定性数据，这里并没有事先进行频数统计
    # 第二个参数为划分的区间个数
    fig, ax = plt.subplots()
    p = ax.hist(heights, 100, range=ranges)  #
    ax.set(xlabel='t', ylabel=ylabel)
    # plt.show()
    fig.savefig(os.path.join(savePath, ylabel + "_hist.png"))
    return p