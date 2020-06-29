#!/usr/bin/env python
# coding: utf-8
import numpy as np
import parl
from parl import layers
from paddle import fluid
from parl.utils import logger,summary
from parl.utils import action_mapping 
from parl.utils import ReplayMemory 
from rlschool import make_env 
from parl.algorithms import DDPG




ACTOR_LR  = 1e-4        # Actor网络更新的 learning rate
CRITIC_LR = 1e-4        # Critic网络更新的 learning rate
GAMMA = 0.99              # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU   = 0.001             # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE        = int(2e6)  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01       # reward 的缩放因子
BATCH_SIZE   = 256        # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 50000000   # 总训练步数
TEST_EVERY_STEPS  = 1e4        # 每个N步评估一下算法效果，每次评估5个episode求平均reward




class ActorModel(parl.Model):
    def __init__(self, act_dim):
        self.fc1 = layers.fc(size=64, act='relu')
        self.fc2 = layers.fc(size=64, act='relu')
        self.fc3 = layers.fc(size=act_dim, act='tanh')

    def policy(self, obs):
        hidden1 = self.fc1(obs)
        hidden2 = self.fc2(hidden1)
        logits  = self.fc3(hidden2)
        return logits




class CriticModel(parl.Model):
    def __init__(self):
        hidden_size = 64
        self.fc1 = layers.fc(size=hidden_size, act='relu')
        self.fc2 = layers.fc(size=hidden_size, act='relu')
        self.fc3 = layers.fc(size=1, act=None)
    def value(self, obs, act):
        concat = layers.concat([obs, act], axis=1)
        hidden1 = self.fc1(concat)
        hidden2 = self.fc2(hidden1)
        Q = self.fc3(hidden2)
        Q = layers.squeeze(Q, axes=[1])
        return Q




class QuadrotorModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()




class QuadrotorAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim=4):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(QuadrotorAgent, self).__init__(algorithm)

        # 注意：最开始先同步self.model和self.target_model的参数.
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        act = np.squeeze(act)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost




def run_episode(env, agent, rpm):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        action = np.clip(np.random.normal(action, 1.0), -1.0, 1.0)
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])
        next_obs, reward, done, info = env.step(action)
        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs,                     batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.squeeze(action)
            action = np.clip(action, -1.0, 1.0)
            action = action_mapping(action, env.action_space.low[0], 
                                    env.action_space.high[0])
            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            steps += 1

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)




# 创建飞行器环境
env = make_env("Quadrotor", task="velocity_control",seed=0)
env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]


# 根据parl框架构建agent
model = QuadrotorModel(act_dim)
algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
agent = QuadrotorAgent(algorithm, obs_dim, act_dim)
# agent.restore('./model_dir/Pre_Training.ckpt')

rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)




#启动训练
test_flag = 0
total_steps = 0
while total_steps < TRAIN_TOTAL_STEPS:
    train_reward, steps = run_episode(env, agent, rpm)
    total_steps += steps
    summary.add_scalar('train/episode_reward', train_reward, total_steps)

    #logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward)) # 打印训练reward
    if total_steps // TEST_EVERY_STEPS >= test_flag: # 每隔一定step数，评估一次模型
        while total_steps // TEST_EVERY_STEPS >= test_flag:
                test_flag += 1
        evaluate_reward = evaluate(env, agent)
        logger.info('Steps {}, Test reward: {}'.format(total_steps, evaluate_reward)) # 打印评估的reward
        summary.add_scalar('eval/episode_reward', evaluate_reward, total_steps)

        # 每评估一次，就保存一次模型，以训练的step数命名
        ckpt = 'star3_model_dir/steps_{}.ckpt'.format(total_steps)
        agent.save(ckpt)

#
# ckpt = 'model_dir/steps_1000000.ckpt'  # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称
#
# agent.restore(ckpt)
# evaluate_reward = evaluate(env, agent)
# logger.info('Evaluate reward: {}'.format(evaluate_reward)) # 打印评估的reward

