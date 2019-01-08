import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO

env = gym.make('CartPole-v1')  # CartPole连杆游戏
env = DummyVecEnv([lambda: env])

model = TRPO(MlpPolicy, env, verbose=1)  # 使用全连接网络模型
model.learn(total_timesteps=25000)  # 训练
model.save("trpo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)  # 预测
    obs, rewards, dones, info = env.step(action)  # 运行
    env.render()  #绘制
