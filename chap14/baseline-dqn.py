from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN

env = make_atari('BreakoutNoFrameskip-v4')  # Atari模型

model = DQN(CnnPolicy, env, verbose=1)  # 使用CnnPolicy建模Deep Q Learning Networks
model.learn(total_timesteps=25000)  # 训练
model.save("dpn")  # 保存模型

obs = env.reset()  # 重置环境
while True:
    action, _states = model.predict(obs)  # 预测
    obs, rewards, dones, info = env.step(action)  # 进行下一步游戏
    env.render()  # 绘图
