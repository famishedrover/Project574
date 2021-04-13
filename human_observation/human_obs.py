# combines classifiers and dfa code.

from human_observation.read_models import RunClassifier 
from ltl_2_dfa import DFA_Graph 


class DFAWrapper() : 

	def __init__(self, path, reward):
		self.terminal_reward = reward 
		self.NEVER_CLAIM_PATH = path 
		self.dfa = DFA_Graph.DFA(self.NEVER_CLAIM_PATH, self.terminal_reward)
		self.classifier = RunClassifier()


	def get_states_count(self):
		return len(self.dfa.G.nodes)

	def get_dfa_state(self,image):
		# Transition on this image. 
		prediction_list, prediction_confidence = self.classifier.make_prediction(image)
		self.dfa.transition(prediction_list, prediction_confidence)
		return self.dfa.current_state

	def get_current_state(self):
		return self.dfa.current_state

	def get_reward(self):
		return self.dfa.get_reward()


	def reset(self): 
		self.dfa.reset()
		# self.classifier = RunClassifier()