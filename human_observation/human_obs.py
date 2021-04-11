# combines classifiers and dfa code.

from human_observation.read_models import RunClassifier 
from ltl_2_dfa import DFA_Graph 


class DFAWrapper() : 

	def __init__(self, path):
		self.NEVER_CLAIM_PATH = path 
		self.dfa = DFA_Graph.DFA(self.NEVER_CLAIM_PATH)
		self.classifier = RunClassifier()


	def get_states_count(self):
		return len(self.dfa.G.nodes)

	def transition_on_image(self,image):
		prediction_list = self.classifier.make_prediction(image)
		self.dfa.transition(prediction_list)
		return self.dfa.current_state

	def get_current_state(self):
		return self.dfa.current_state

	def get_reward(self):
		return self.dfa.get_reward()


