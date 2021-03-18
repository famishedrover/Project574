from ltlf2dfa.parser.pltlf import PLTLfParser
from graphviz import Source

import pygraphviz
from networkx.drawing import nx_agraph


FORMULA = "H(a -> Y b)"



parser = PLTLfParser()
formula_str = FORMULA
formula = parser(formula_str)       # returns a PLTLfFormula

print(formula)                      # prints "H(a -> Y (b))"

dfa = formula.to_dfa()

# dfa.replace("\n", "")


def convertToGraph(dfa):
	G = nx_agraph.from_agraph(pygraphviz.AGraph(dfa))

	return G

def view_graph(dfa):
	s = Source(dfa, filename="dfa.gv", format="png")
	s.view()


view_graph(dfa)



G = convertToGraph(dfa)

print (type(G))

# to print nodes
print (G.nodes)

# to print edge labels
print(G.edges(data=True))

# use networkx to convert dfa dot to networkx graph so that it can be later modified. 




