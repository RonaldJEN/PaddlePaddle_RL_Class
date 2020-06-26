#!/usr/bin/env python
# coding: utf-8

'''
【作业3评分标准】（需要保留notebook上每个cell运行之后的log信息）：
1、完全没有log信息，59分
2、完成部分函数的编写，有部分log信息，但代码运行失败，70分
3、代码运行成功，但test_reward无法收敛（所有分数均低于-140），80分
4、代码运行成功，分数有上涨趋势，且最终输出的test_reward收敛至[-120, -140]之间，90分
5、代码运行成功，分数有上涨趋势，且最终输出的test_reward收敛至大于-120，100分

【作业3满分标准补充说明】
1、最后10个分数一半以上大于-120，100分
2、如果最后10个分数表现不好，但是最近50个分数大部分都大于-120，100分
3、其他情况酌情给分
'''

import parl
from parl import layers
import paddle.fluid as fluid
import numpy as np
import gym
from parl.utils import logger
from parl.algorithms import DQN
import random
import collections

LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE= int(20000)    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200   # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 64   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等

LEARNING_RATE = 5e-4 # 学习率




class Model(parl.Model):
    def __init__(self, act_dim):
        hidden_size = 256
        self.net1 = layers.fc(size=hidden_size, act='relu')
        self.net2 = layers.fc(size=hidden_size, act='relu')
        self.net3 = layers.fc(size=hidden_size, act='relu')
        self.net4 = layers.fc(size=hidden_size, act='relu')
        self.net5 = layers.fc(size=act_dim, act=None)
    def value(self, obs):
        hidden1  = self.net1(obs)
        hidden2  = self.net2(hidden1)
        hidden3  = self.net3(hidden2)
        hidden4  = self.net4(hidden3)
        Q_value  = self.net5(hidden4)
        return Q_value




class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.001, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost




class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)
    def append(self, exp):
        self.buffer.append(exp)
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
        return np.array(obs_batch).astype('float32'),             np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')
    def __len__(self):
        return len(self.buffer)




# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)




# 创建环境
env = gym.make('MountainCar-v0')
action_dim = env.action_space.n  # MountainCar-v0: 3
obs_shape  = env.observation_space.shape  # MountainCar-v0: (2,)

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

# 根据parl框架构建agent
# 4. 请参考课堂Demo，嵌套Model, DQN, Agent构建 agent
model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(algorithm,obs_dim=obs_shape[0],act_dim=action_dim,e_greed=0.2,  e_greed_decrement=1e-6)  


# 加载模型
# save_path = './dqn_model.ckpt'
# agent.restore(save_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm)

max_episode = 20000

# 开始训练
episode = 0
while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    for i in range(0, 50):
        total_reward = run_episode(env, agent, rpm)
        episode += 1

    # test part
    eval_reward = evaluate(env, agent, render=False)  # render=True 查看显示效果
    logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
        episode, agent.e_greed, eval_reward))

# 训练结束，保存模型
save_path = './model_dir/dqn_model.ckpt'
agent.save(save_path)

