from gym_minigrid.wrappers import *
from human_observation.human_obs import DFAWrapper

import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch 

class DFAEnvWrapper(RGBImgObsWrapper):
	def __init__(self, env, dfas, step_cost=None, env_terminal_reward=1):
		super().__init__(env)
		self.env = env
		self.dfas = dfas
		self.step_cost = step_cost
		self.env_terminal_reward = env_terminal_reward
		
	def step(self, action):
		next_state, reward, done, info = self.env.step(action)
		reward *= self.env_terminal_reward

		next_q_states = []
		for ix in range(len(self.dfas)) : 
			next_q = self.dfas[ix].get_dfa_state(next_state['image'])
			next_r_dfa = self.dfas[ix].get_reward()
			reward += next_r_dfa

			q_onehot = torch.zeros(self.dfas[ix].get_states_count())
			q_onehot[next_q] = 1

			next_q_states.append(q_onehot)

		next_q_states = torch.hstack(next_q_states)
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

		next_q_states = torch.hstack(next_q_states)
		obs['q'] = next_q_states


		return obs


if __name__ == "__main__":

	n = 6
	env_name = "MiniGrid-Empty-{}x{}-v0".format(n+2,n+2)
	env = gym.make(env_name)
	env = RGBImgObsWrapper(env)

	LTL_PATH = "./ltl_2_dfa/neverClaimFiles/never_claim_6.txt"
	dfa = DFAWrapper(LTL_PATH, 1)
	dfa2 = DFAWrapper(LTL_PATH, 1)


	env = DFAEnvWrapper(env, [dfa, dfa2])





	MAX_STEPS = 50 
	counter = 0 

	while MAX_STEPS : 
		MAX_STEPS -=1
		counter += 1


		env.render()
		action = env.action_space.sample()

		act = input()
		obs,reward,done,info = env.step(int(act))
		print ("DFA : ", obs['q'])

		if done : 
			break

		image = obs['image']



