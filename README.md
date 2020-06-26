# 百度AI Studio 强化学习7日打卡营总结
## 课程Notebook
+ 第一章、强化学习介绍
    + [环境搭建](https://nbviewer.jupyter.org/github/RonaldJEN/PaddlePaddle_RL_Class/blob/master/notebook/lesson1_MakeEnv.ipynb)
+ 第二章、基于表格型方法求解RL
    + [Sarsa](https://nbviewer.jupyter.org/github/RonaldJEN/PaddlePaddle_RL_Class/blob/master/notebook/lesson2.1_Sarsa.ipynb)
    + [Q-learning](https://nbviewer.jupyter.org/github/RonaldJEN/PaddlePaddle_RL_Class/blob/master/notebook/lesson2.2_Qlearning.ipynb)
+ 第三章、基于神经网络方法求解RL
    + [DQN](https://nbviewer.jupyter.org/github/RonaldJEN/PaddlePaddle_RL_Class/blob/master/notebook/lesson3_DQN.ipynb)
+ 第 四章、基于策略梯度求解RL
    + [Policy Gradient](https://nbviewer.jupyter.org/github/RonaldJEN/PaddlePaddle_RL_Class/blob/master/notebook/lesson4_PG.ipynb)
+ 第五章、连续动作空间上求解RL
    + [DDPG](https://nbviewer.jupyter.org/github/RonaldJEN/PaddlePaddle_RL_Class/blob/master/notebook/lesson5_FinalHW.ipynb)

## 大作业完成心得
题目：四轴飞行器悬停任务
![image](https://github.com/RonaldJEN/PaddlePaddle_RL_Class/blob/master/pic/1.jpg)
四轴飞行器状态简介：
![image](https://github.com/RonaldJEN/PaddlePaddle_RL_Class/blob/master/pic/2.jpg)
初始模型（动作固定）：
可以看到当四个螺旋发动机电压值固定时，动作[1.0, 1.0, 1.0, 1.0]，将使无初速度的飞行器垂直向上或向下运动。
![image](https://github.com/RonaldJEN/PaddlePaddle_RL_Class/blob/master/pic/3.gif)
训练trick总结：
1.多轮迭代，按训练的情况手动调整学习率

第一轮训练，经过八十万个step,回报终于从-8000多变成正数
![image](https://github.com/RonaldJEN/PaddlePaddle_RL_Class/blob/master/pic/4.jpg)
第N轮训练，当效果好时降低学习率。到了后期回报稳定在八千上下
![image](https://github.com/RonaldJEN/PaddlePaddle_RL_Class/blob/master/pic/5.jpg)

2.将Actor的输出动作从四个旋翼的电压尽量相似，在前期训练时新增一个调整项，用调整项对四个旋翼的电压做修正，使得4个旋翼的最终电压差异不会太大

最后八千分的交互范例如下，可以看到相比初始固定模型，此时模型已能较快达到悬停。
![image](https://github.com/RonaldJEN/PaddlePaddle_RL_Class/blob/master/pic/6.)

