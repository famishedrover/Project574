from DQN.dqnagent import Agent
from collections import deque

import torch
from human_observation.human_obs import DFAWrapper
from gym_minigrid.wrappers import *
from DQN.EnvDFAWrapper import DFAEnvWrapper

from matplotlib import pyplot as plt 


LTL_PATH = "./ltl_2_dfa/neverClaimFiles/never_claim_7.txt"
dfa = DFAWrapper(LTL_PATH, reward=5, low_reward=1)
LTL_PATH = "./ltl_2_dfa/neverClaimFiles/never_claim_7.txt"
dfa2 = DFAWrapper(LTL_PATH, reward=5, low_reward=1)


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






# action_size = env.action_space.n
action_size = 3


seed = 1

agent = Agent((h,w),action_size, seed, dfas_size)






NUM_EPISODES = 10 

eps = 0.01
eps_decay = 0.996
eps_end = 0.001

agent.read_model("./DQN/models/dfa_7/checkpoint7.pth")

for i in range(NUM_EPISODES) : 

	state = env.reset()
	done = False 
	while not done : 
		env.render()
		action = agent.act(state,eps)
		# action = int(input())
		next_state,reward,done,_ = env.step(action)

		# agent.step(state,action,reward,next_state,done)

		state = next_state
















