import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import Env  # 替换为你的环境文件和类
env = Env()

check_env(env, warn=True)