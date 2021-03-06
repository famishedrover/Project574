import networkx as nx 

from ltl_2_dfa.never_claim_reader import NeverClaim 
from ltl_2_dfa.base_graph import BaseGraph

from matplotlib import pyplot as plt 


from networkx.drawing.nx_agraph import to_agraph 
from PIL import Image 
import time

import os 

import human_observation.config as human_obs_config


TEMPORARY_DIR = "./ltl_2_dfa/tmp"

if not os.path.exists(TEMPORARY_DIR):
	os.mkdir(TEMPORARY_DIR)


class DFA (BaseGraph):
	def __init__(self, PATH, reward=1, low_reward=1) : 
		# PATH : NEVERCLAIM PATH 

		self.ncm = NeverClaim(PATH)
		self.G = self.ncm.getnxGraph()

		self.init_node = self.getInitNode()
		self.current_state = self.init_node

		self.NX_SEED = 100

		self.counter = 0

		self.terminal_reward = reward
		self.low_reward = low_reward



	def reset(self):
		self.counter = 0
		self.current_state = self.init_node
		

	def parse_formula_for_fluents(formula):
		pass


	def transition(self, s, prediction_confidence = None):

		self.current_reward = 0

		edges = self.G.edges(self.current_state, data=True)

		true_transitions = []
		for e in edges : 
			evaluation = eval(e[2]["data"])
			# print ("Evaluation for", e[2]["data"], " is ",evaluation)
			if(evaluation) : 
				# next node.
				true_transitions.append([e[1], e[2]["data"]])

		# if multiple transitions are possible, pick the first one. 
		# Make the transition.
		assert len(true_transitions)>0, "DFA Found no valid transitions"

		# pick one transition out of true_transitions. 

		if(len(true_transitions) != 1) :  
			# pick state without true.
			true_transition_idx = -1
			for ix in range(len(true_transitions)) :
				if("true" in true_transitions[ix][1]) : 
					true_transition_idx = ix

			if(true_transition_idx >=0 ):
				del true_transitions[true_transition_idx]

		

		# true_transitions[0][0] is the transition we wish to take. Just count the number of 
		# lows here. 


		
		eval_string = true_transitions[0][1]
		symbol_table = self.ncm.SymbolTable

		this_reward = 0

		for symbol in symbol_table : 
			if symbol in eval_string : 
				# print (symbol, prediction_confidence[symbol])
				if(prediction_confidence[symbol] >= 0.5 and prediction_confidence[symbol] < human_obs_config.THRESHOLD) : 
					this_reward -= self.low_reward

		self.current_reward +=  this_reward

		self.current_state = true_transitions[0][0]
		self.counter += 1	


	def getInitNode(self):
		G = self.G
		for i in G.nodes() : 
			if(G.nodes[i]['type'] == 'i') : 
				return i

	def get_reward(self):
		# states can be "i", "a", "d"
		if self.G.nodes[self.current_state]['type'] == 'a' :
			self.current_reward += self.terminal_reward

		return self.current_reward



	def draw_graph(self):

		G = self.G 

		# print(G.edges)
		# print(G.nodes)

		my_pos = nx.spring_layout(G, seed = 100)

		node_color_map = []
		for node in G : 
			if node == self.current_state : 
				G.nodes[node]['color'] = 'red'
			else : 
				G.nodes[node]['color'] = 'blue'


		# arrow size: '0' makes it look like an indirected graph
		G.graph['edge'] = {'arrowsize': '1', 'splines': 'curved'}
		G.graph['graph'] = {'scale': '4', 'ranksep':'2'}


		A = to_agraph(G)
		A.layout('dot')

		# set edge labels
		for pair in G.edges(data=True):
		    edge = A.get_edge(pair[0], pair[1])
		    edge.attr['label'] = "\n" + pair[2]['data']
		    edge.attr['fontsize'] = 8




		FILE_NAME = str(self.counter) + '.png'
		FILE_NAME = os.path.join(TEMPORARY_DIR, FILE_NAME)
		A.draw(FILE_NAME)




if __name__ == "__main__" : 
	dfa = DFA("./ltl_2_dfa/neverClaimFiles/never_claim_3.txt")
	G = dfa.G 
	print(G.nodes(data=True))


	# dfa.view_graph()

	state = []

	state.append({
	'left' : True,
	'bottom' : False,
	'true' : True
	}) 
	state.append({
	'left' : False,
	'bottom' : True,
	'true' : True
	}) 
	state.append({
	'left' : True,
	'bottom' : True,
	'true' : True
	}) 

	dfa.transition(state[0])
	dfa.draw_graph()
	print(dfa.current_state)
	dfa.transition(state[1])
	dfa.draw_graph()
	print(dfa.current_state)
	dfa.transition(state[2])
	print(dfa.current_state)
	dfa.draw_graph()



	print (G.edges())




