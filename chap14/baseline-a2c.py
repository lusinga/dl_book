import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

n_cpu = 4  # 支持4线程
env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])

model = A2C(MlpPolicy, env, verbose=1)  # 使用MlpPolicy的A2C算法
model.learn(total_timesteps=25000)  # 训练

obs = env.reset()
while True:
    action, _states = model.predict(obs)  # 预测
    obs, rewards, dones, info = env.step(action)  # 执行一步游戏
    env.render()  # 显示
