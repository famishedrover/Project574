import networkx as nx


from networkx.drawing.nx_agraph import write_dot
from graphviz import Source




class BaseGraph :
	def __init__():
		pass

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