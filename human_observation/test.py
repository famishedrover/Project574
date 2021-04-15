from human_observation.human_obs import DFAWrapper
from gym_minigrid.wrappers import *

LTL_PATH = "./ltl_2_dfa/neverClaimFiles/never_claim_7.txt"

dfa = DFAWrapper(LTL_PATH, 1)
n = 6

env_name = "MiniGrid-Empty-{}x{}-v0".format(n+2,n+2)
env = gym.make(env_name)
env = RGBImgObsWrapper(env)
env.reset()



def simulate():
	MAX_STEPS = 300

	counter = 0


	while MAX_STEPS : 
		MAX_STEPS -=1
		counter += 1

		print(counter, )


		action = env.action_space.sample()

		# print (env.action_space)

		# act = input()
		act = action
		obs,reward,done,info = env.step(int(act))

		if done : 
			break

		image = obs['image']
		# print (image.shape)

		# get_dfa_state

		q_next_state = dfa.get_dfa_state(image)
		reward = dfa.get_reward()

		print (q_next_state)

		# env.render()



		# print (q_next_state, reward)
		# print (dfa.classifier.prediction_dict)


simulate()

dfa.reset()
print (dfa.dfa.current_state)

print ("RESETTING")

simulate()