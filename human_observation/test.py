from human_observation.human_obs import DFAWrapper
from gym_minigrid.wrappers import *

LTL_PATH = "./ltl_2_dfa/neverClaimFiles/never_claim_4.txt"

dfa = DFAWrapper(LTL_PATH, 1)
n = 6

env_name = "MiniGrid-Empty-{}x{}-v0".format(n+2,n+2)
env = gym.make(env_name)
env = RGBImgObsWrapper(env)
env.reset()



MAX_STEPS = 50

while MAX_STEPS : 
	MAX_STEPS -=1


	action = env.action_space.sample()

	print (env.action_space)

	act = input()
	obs,reward,done,info = env.step(int(act))

	image = obs['image']
	print (image.shape)

	# get_dfa_state

	q_next_state = dfa.get_dfa_state(image)
	reward = dfa.get_reward()

	env.render()



	print (q_next_state, reward)
	print (dfa.classifier.prediction_dict)

