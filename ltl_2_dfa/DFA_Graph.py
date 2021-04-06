import networkx as nx 

from never_claim_reader import NeverClaim 
from base_graph import BaseGraph



class DFA (BaseGraph):
	def __init__(self, PATH) : 
		# PATH : NEVERCLAIM PATH 
		self.ncm = NeverClaim(PATH)
		self.G = self.ncm.getnxGraph()

		self.init_node = self.getInitNode()
		self.current_state = self.init_node




	def transition(self, s):
		edges = self.G.edges(self.current_state, data=True)

		true_transitions = []
		for e in edges : 
			evaluation = eval(e[2]["data"])
			# print ("Evaluation for", e[2]["data"], " is ",evaluation)
			if(evaluation) : 
				# next node.
				true_transitions.append(e[1])

		# if multiple transitions are possible, pick the first one. 
		# Make the transition.
		assert len(true_transitions)>0, "DFA Found no valid transitions"

		self.current_state = true_transitions[0]


	def getInitNode(self):
		G = self.G
		for i in G.nodes() : 
			if(G.nodes[i]['type'] == 'i') : 
				return i


if __name__ == "__main__" : 
	dfa = DFA("never_claim_3.txt")
	G = dfa.G 
	print(G.nodes(data=True))


	dfa.view_graph()

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
	print(dfa.current_state)
	dfa.transition(state[1])
	print(dfa.current_state)
	dfa.transition(state[2])
	print(dfa.current_state)


