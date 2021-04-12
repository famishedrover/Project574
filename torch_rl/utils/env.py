import gym
from gym_minigrid.wrappers import  *


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env = RGBImgObsWrapper(env)
    # env = ImgObsWrapper(env)
    env.seed(seed)
    return env
