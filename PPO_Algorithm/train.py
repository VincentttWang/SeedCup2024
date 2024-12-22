import os
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

from env import Env  # 导入你的环境


def train():
    # 创建环境
    env = Env()
    env = Monitor(env)  # 监控环境，记录日志

    policy_kwargs = dict(
    net_arch=[64, 64],  # 两层隐藏层，每层 64 个神经元（默认是更大的结构）
    )

    
    if os.path.exists("ppo_model.zip"):
    # 定义 PPO 算法的模型
        print("load model")
        model = PPO.load("ppo_model", env, policy_kwargs=policy_kwargs, verbose=1,batch_size=256,n_steps = 1024,
                         device="mps",learning_rate=0.0003)
    else:
        print("create model")
        model = PPO("MlpPolicy", env, verbose=1,batch_size=256,n_steps = 1024,device="mps",
                    learning_rate=0.0003,policy_kwargs=policy_kwargs)
        

    # 可选：使用 EvalCallback 来定期评估模型性能
    eval_callback = EvalCallback(env, best_model_save_path="./models/",
                                 log_path="./logs/", eval_freq=5096,
                                 deterministic=True, render=False)

    # 训练模型
    model.learn(total_timesteps=100000, callback=eval_callback)

    # 保存训练好的模型
    model.save("ppo_model")

    print("Training complete!")

if __name__ == "__main__":
    train()