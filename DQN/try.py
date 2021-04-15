from DQN.dqnagent import Agent
from collections import deque

import torch
from human_observation.human_obs import DFAWrapper
from gym_minigrid.wrappers import *
from DQN.EnvDFAWrapper import DFAEnvWrapper

from matplotlib import pyplot as plt 

LTL_PATH = "./ltl_2_dfa/neverClaimFiles/never_claim_5.txt"
dfa = DFAWrapper(LTL_PATH, reward=5)
LTL_PATH = "./ltl_2_dfa/neverClaimFiles/never_claim_5.txt"
dfa2 = DFAWrapper(LTL_PATH, reward=5)


n = 6

env_name = "MiniGrid-Empty-{}x{}-v0".format(n+2,n+2)
env = gym.make(env_name)
env = RGBImgObsWrapper(env)
env = DFAEnvWrapper(env, [dfa, dfa2], step_cost=-0.1, env_terminal_reward=100)

obs = env.reset()

import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




h,w,_ = obs['image'].shape
dfas_size = obs['q'].shape[0]
print ("DFAS SIZE = ", dfas_size)

# print (env.unwrapped())

print (env.agent_pos)