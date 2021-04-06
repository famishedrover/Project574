import networkx as nx 

from never_claim_reader import NeverClaim 

from networkx.drawing.nx_agraph import write_dot
from graphviz import Source

class DFA :
	def __init__(self, PATH) : 
		# PATH : NEVERCLAIM PATH 
		self.ncm = NeverClaim(PATH)
		self.G = self.ncm.getnxGraph()

	def view_graph(self):
		# Need to separate view_graph function and Improve drawing using matplotlib.
		dfa = self.getdotfile()
		s = Source(dfa, filename="dfa.gv", format="png")
		s.view()

	def getdotfile(self):
		write_dot(self.G, "tx.dot")
		with open("tx.dot", "r") as f :
			x = f.read()
		return x 


if __name__ == "__main__" : 
	dfa = DFA("never_claim_2.txt")
	G = dfa.G 
	print(G.nodes(data=True))

	dfa.view_graph()
