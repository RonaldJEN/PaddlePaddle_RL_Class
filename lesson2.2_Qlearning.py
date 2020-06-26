#!/usr/bin/env python
# coding: utf-8
'''
迷宫游戏
作业2评分标准（需要保留notebook上每个cell运行之后的log信息）：
1、完全没有log信息，59分
2、完成部分函数的编写，有部分log信息，但代码运行失败，70分
3、代码运行成功，但最终输出的test reward无法收敛至1.0，80分
4、代码运行成功，且最终输出的test reward收敛至1.0，100分
'''
import gym
import numpy as np

class QLearningAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n      # 动作维度，有几个动作可选
        self.lr = learning_rate # 学习率
        self.gamma = gamma      # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        # 1. 请完成sample函数功能
        if np.random.uniform() < (1-self.epsilon):
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        # 2. 请完成predict函数功能
        Q_array     = self.Q[obs, :]
        Maximize_Q  = np.max(Q_array)
        action_array= np.where(Q_array==Maximize_Q)
        action_list = action_array[0]
        return np.random.choice(action_list,replace=True, p=None)

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, done):
        """ off-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            done: episode是否结束
        """
        # 3. 请完成learn函数功能（Q-learning）
        q_predict = self.Q[obs, action]
        if done:
            q_target = reward  
        else:
            q_target = reward + self.gamma * self.Q[next_obs, :].max()
        self.Q[obs, action] += self.lr * (q_target - q_predict)  
    # 保存Q表格数据到文件
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')
    
    # 从文件中读取数据到Q表格中
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')


def run_episode(env, agent, render=False):
    total_steps = 0 # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset() # 重置环境, 重新开一局（即开始新的一个episode）

    while True:
        action = agent.sample(obs) # 根据算法选择一个动作
        next_obs, reward, done, _ = env.step(action) # 与环境进行一个交互
        # 训练 Q-learning算法
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1 # 计算step数
        if render:
            env.render() #渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps

def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs) # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


#使用gym创建迷宫环境，设置is_slippery为False降低环境难度
env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up

# 创建一个agent实例，输入超参数
agent = QLearningAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.001,
        gamma=0.99,
        e_greed=0.05)

# 训练500个episode，打印每个episode的分数
for episode in range(5000):
    ep_reward, ep_steps = run_episode(env, agent, False)
    print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

# 全部训练结束，查看算法效果
test_reward = test_episode(env, agent)
print('test reward = %.1f' % (test_reward))

