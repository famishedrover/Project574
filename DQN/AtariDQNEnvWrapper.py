# from gym_minigrid.wrappers import *
from human_observation.human_obs import DFAWrapper

import gym
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch 

class AtariDQNEnvWrapper(gym.Wrapper):
	def __init__(self, env, dfas, step_cost=None, env_terminal_reward=1):
		super().__init__(env)
		self.env = env
		self.dfas = dfas
		self.step_cost = step_cost
		self.env_terminal_reward = env_terminal_reward
		
		
	def step(self, action):
		next_state_, reward, done, info = self.env.step(action)
		reward *= self.env_terminal_reward

		next_state = {}
		next_state["image"] = next_state_


		next_q_states = []
		for ix in range(len(self.dfas)) : 
			next_q = self.dfas[ix].get_dfa_state(next_state['image'])
			next_r_dfa = self.dfas[ix].get_reward()
			reward += next_r_dfa

			q_onehot = torch.zeros(self.dfas[ix].get_states_count())
			q_onehot[next_q] = 1

			next_q_states.append(q_onehot)

		# next_q_states = torch.hstack(next_q_states)
		next_q_states = torch.cat(next_q_states, dim=0)
		next_state['q'] = next_q_states

		if self.step_cost : 
			reward += self.step_cost

		return next_state, reward, done, info
	
	def reset(self):
		obs = self.env.reset()

		next_q_states = []
		for ix in range(len(self.dfas)):
			self.dfas[ix].reset()

			next_q = self.dfas[ix].get_current_state()
			q_onehot = torch.zeros(self.dfas[ix].get_states_count())
			q_onehot[next_q] = 1
			next_q_states.append(q_onehot)

		# next_q_states = torch.hstack(next_q_states)
		next_q_states = torch.cat(next_q_states, dim=0)

		dict_obs = {}
		dict_obs["image"] = obs
		dict_obs["q"] = next_q_states


		return dict_obs


if __name__ == "__main__":

	n = 6
	env_name = "Breakout-v0"
	env = gym.make(env_name)
	# env = RGBImgObsWrapper(env)

	LTL_PATH = "./ltl_2_dfa/neverClaimFiles/never_claim_6.txt"
	dfa = DFAWrapper(LTL_PATH, 5, low_reward=2)
	dfa2 = DFAWrapper(LTL_PATH, 4, low_reward=2)


	env = AtariDQNEnvWrapper(env, [dfa, dfa2], env_terminal_reward=100)

	env.reset()




	MAX_STEPS = 50 
	counter = 0 

	while MAX_STEPS : 
		MAX_STEPS -=1
		counter += 1


		# env.render()
		action = env.action_space.sample()

		# act = input()
		act = action
		obs,reward,done,info = env.step(int(act))
		# print ("DFA : ", obs['q'])
		# print ("reward", reward)

		if done : 
			break

		image = obs['image']



