#!/usr/bin/env python
# coding: utf-8


'''
PG解决Pong
作业4 PolicyGradient作业、评分标准（需要保留notebook上每个cell运行之后的log信息）：
1、完全没有log信息，59分
2、完成部分函数的编写，有部分log信息，但代码运行失败，70分
3、代码运行成功，但Test reward收敛效果不好（所有分数均低于0），80分
4、代码运行成功，分数有上涨趋势，且最后（Episode 2800、2900、3000）输出的3个Test reward的最高分在[0, 10]之间，90分
5、代码运行成功，分数有上涨趋势，且最后（Episode 2800、2900、3000）输出的3个Test reward的最高分大于10，100分

【作业4满分标准补充说明】
1、最后10个分数一半以上大于10，100分
2、如果最后10个分数表现不好，但是最近50个分数大部分都大于10，100分
3、其他情况酌情给分
'''

import gym
import numpy as np
from parl.algorithms import PolicyGradient 
import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger




LEARNING_RATE = 5e-4




class Model(parl.Model):
    def __init__(self, act_dim):
        # 2. 请参考课程Demo，配置model结构
        act_dim = act_dim
        hidden_size = act_dim * 24
        self.net1 = layers.fc(size=hidden_size, act='tanh')
        self.net2 = layers.fc(size=act_dim, act='softmax')
    def forward(self, obs):  # 可直接用 model = Model(5); model(obs)调用
        # 3. 请参考课程Demo，组装policy网络
        hidden1  = self.net1(obs)
        out      = self.net2(hidden1)
        return out



class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(
                self.learn_program):  # 搭建计算图用于 更新policy网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)  # 增加一维维度
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
        act = np.random.choice(range(self.act_dim), p=act_prob)  # 根据动作概率选取动作
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost




def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs = preprocess(obs) # from shape (210, 160, 3) to (100800,)
        obs_list.append(obs)
        action = agent.sample(obs) # 采样动作
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = preprocess(obs) # from shape (210, 160, 3) to (100800,)
            action = agent.predict(obs) # 选取最优动作
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)




# Pong 图片预处理
def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195] # 裁剪
    image = image[::2,::2,0] # 下采样，缩放2倍
    image[image == 144] = 0 # 擦除背景 (background type 1)
    image[image == 109] = 0 # 擦除背景 (background type 2)
    image[image != 0] = 1 # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float).ravel()


# 根据一个episode的每个step的reward列表，计算每一个Step的Gt
def calc_reward_to_go(reward_list, gamma=0.99):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr


# 创建环境
env = gym.make('Pong-v0')
obs_dim = 80 * 80
act_dim = env.action_space.n
logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

# 根据parl框架构建agent
# 4. 请参考课堂Demo构建 agent，嵌套Model, PolicyGradient, Agent
#
model = Model(act_dim=act_dim)
alg = PolicyGradient(model, lr=LEARNING_RATE)
agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)


# 加载模型
# if os.path.exists('./model.ckpt'):
#     agent.restore('./model.ckpt')

for i in range(20000):
    obs_list, action_list, reward_list = run_episode(env, agent)
    # if i % 10 == 0:
    #     logger.info("Train Episode {}, Reward Sum {}.".format(i, 
    #                                         sum(reward_list)))

    batch_obs = np.array(obs_list)
    batch_action = np.array(action_list)
    batch_reward = calc_reward_to_go(reward_list)

    agent.learn(batch_obs, batch_action, batch_reward)
    if (i + 1) % 100 == 0:
        total_reward = evaluate(env, agent, render=False)
        logger.info('Episode {}, Test reward: {}'.format(i + 1, 
                                            total_reward))

# save the parameters to ./model.ckpt
agent.save('./model.ckpt')

