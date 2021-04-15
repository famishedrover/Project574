from DQN.dqnagent import Agent
from collections import deque

import torch
from human_observation.human_obs import DFAWrapper
from gym_minigrid.wrappers import *
from DQN.EnvDFAWrapper import DFAEnvWrapper

from matplotlib import pyplot as plt 

LTL_PATH = "./ltl_2_dfa/neverClaimFiles/never_claim_7.txt"
dfa = DFAWrapper(LTL_PATH, reward=5, low_reward = 2)
LTL_PATH = "./ltl_2_dfa/neverClaimFiles/never_claim_7.txt"
dfa2 = DFAWrapper(LTL_PATH, reward=5, low_reward = 2)


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

def dqn(n_episodes= 200, max_t = 256, eps_start=1.0, eps_end = 0.01,
	   eps_decay=0.996):
	"""Deep Q-Learning
	
	Params
	======
		n_episodes (int): maximum number of training epsiodes
		max_t (int): maximum number of timesteps per episode
		eps_start (float): starting value of epsilon, for epsilon-greedy action selection
		eps_end (float): minimum value of epsilon 
		eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon
		
	"""
	scores = [] # list containing score from each episode
	scores_window = deque(maxlen=100) # last 100 scores
	eps = eps_start
	for i_episode in range(1, n_episodes+1):
		state = env.reset()
		score = 0
		for t in range(max_t):
			env.render()
			action = agent.act(state,eps)
			next_state,reward,done,_ = env.step(action)
			agent.step(state,action,reward,next_state,done)

			state = next_state
			score += reward
			if done:
				print (" Done in steps ", t)
				break
			scores_window.append(score) ## save the most recent score
			scores.append(score) ## sae the most recent score
			eps = max(eps*eps_decay,eps_end)## decrease the epsilon
			# print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
			# if i_episode %1==0:
				# print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))


		if score>=0.0:
			# print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
			torch.save(agent.qnetwork_local.state_dict(),'./DQN/models/checkpoint'+str(i_episode)+'.pth')


		print('\rEpisode {}\tEpisode Score {:.2f}'.format(i_episode,score))

	return scores

scores= dqn()

#plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)),scores)
plt.ylabel('Score')
plt.xlabel('Epsiode #')
plt.show()