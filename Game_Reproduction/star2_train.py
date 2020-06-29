import gym
import numpy as np
from parl import layers
from paddle import fluid
import parl
from parl.utils import logger, summary,ReplayMemory
from parl.algorithms import SAC
import argparse
ACTOR_LR  = 1e-4
CRITIC_LR = 1e-4
gamma     = 0.99
TAU       = 0.005
replay_size = int(2e6)
start_steps = 1e4
REWARD_SCALE = 5.0      # reward 的缩放因子

batch_size = 32
seed   = 0
LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0
class ActorModel(parl.Model):
    def __init__(self, act_dim):
        self.fc1 = layers.fc(size=400, act='relu')
        self.fc2 = layers.fc(size=300, act='relu')
        self.mean_linear    = layers.fc(size=act_dim)
        self.log_std_linear = layers.fc(size=act_dim,act='tanh')
    def policy(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        mu      = self.mean_linear(hid2)
        log_std = self.log_std_linear(hid2)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        log_std = layers.exp(log_std)
        return mu, log_std
class CriticModel(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(size=400, act='relu')
        self.fc2 = layers.fc(size=300, act='relu')
        self.fc3 = layers.fc(size=1, act=None)
        self.fc4 = layers.fc(size=400, act='relu')
        self.fc5 = layers.fc(size=300, act='relu')
        self.fc6 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        hid1 = self.fc1(obs)
        concat1 = layers.concat([hid1, act], axis=1)
        Q1 = self.fc2(concat1)
        Q1 = self.fc3(Q1)
        Q1 = layers.squeeze(Q1, axes=[1])

        hid2 = self.fc4(obs)
        concat2 = layers.concat([hid2, act], axis=1)
        Q2 = self.fc5(concat2)
        Q2 = self.fc6(Q2)
        Q2 = layers.squeeze(Q2, axes=[1])

        return Q1, Q2
class BipedalWalkerAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(BipedalWalkerAgent, self).__init__(algorithm)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program   = fluid.Program()
        self.sample_program = fluid.Program()
        self.learn_program  = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.sample_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.sample_act, _ = self.alg.sample(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.critic_cost, self.actor_cost = self.alg.learn(
                obs, act, reward, next_obs, terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        return act

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            self.sample_program,
            feed={'obs': obs},
            fetch_list=[self.sample_act])[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        [critic_cost, actor_cost] = self.fluid_executor.run(
            self.learn_program,
            feed=feed,
            fetch_list=[self.critic_cost, self.actor_cost])
        self.alg.sync_target()
        return critic_cost[0], actor_cost[0]


def run_train_episode(env, agent, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)

        if rpm.size() < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.sample(batch_obs.astype('float32'))
            action = np.squeeze(action)
        next_obs, reward, done, info = env.step(action)
        if done :
            #reward -100的突变对值函数学习不利，这里把它clip为0
            reward =0
        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > start_steps:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(batch_size)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps

def run_evaluate_episode(env, agent):
    obs = env.reset()
    total_reward = 0
    while True:
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)

        next_obs, reward, done, info = env.step(action)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward

def main():
    env = gym.make(args.env)
    env.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    algorithm = SAC(ActorModel(act_dim),CriticModel(),
        max_action=act_limit,gamma=gamma,tau=TAU,
        actor_lr=ACTOR_LR,critic_lr=CRITIC_LR)
    agent = BipedalWalkerAgent(algorithm, obs_dim, act_dim)
    rpm   = ReplayMemory(replay_size, obs_dim, act_dim)

    test_flag = 0
    total_steps = 0
    while total_steps < args.train_total_steps:
        train_reward, steps = run_train_episode(env, agent, rpm)
        total_steps += steps
        # logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward))
        summary.add_scalar('train/episode_reward', train_reward, total_steps)

        if total_steps // args.test_every_steps >= test_flag:
            while total_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            evaluate_reward = run_evaluate_episode(env, agent)
            logger.info('Steps {}, Evaluate reward: {}'.format(
                total_steps, evaluate_reward))
            summary.add_scalar('eval/episode_reward', evaluate_reward,total_steps)
            ckpt = 'star2_model_dir/steps_{}.ckpt'.format(total_steps)
            agent.save(ckpt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment name', default='BipedalWalkerHardcore-v2')
    parser.add_argument('--train_total_steps',type=int,default=int(50000000),help='maximum training steps')
    parser.add_argument('--test_every_steps',type=int,default=int(1e4),help='the step interval between two consecutive evaluations')
    parser.add_argument('--alpha',type=float,default=0.1,help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
    args = parser.parse_args()
    main()
