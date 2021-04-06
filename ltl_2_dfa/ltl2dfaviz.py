from ltlf2dfa.parser.pltlf import PLTLfParser
from graphviz import Source

import pygraphviz
from networkx.drawing import nx_agraph

import networkx as nx

from networkx.drawing.nx_agraph import write_dot



from matplotlib import pyplot as plt 

FORMULA = "O(a -> Y b)"

# FORMULA = "G ( a )"



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


def plotnxgraph(G):
	nx.draw(G, with_labels = True) 
	plt.savefig("tmp.png")


def getdotfile(G):
	write_dot(G, "tx.dot")
	with open("tx.dot", "r") as f :
		x = f.read()

	print ("DOT FILE \n")


	return x 

# view_graph(dfa)

G = convertToGraph(dfa)

print (type(G))

# to print nodes
print (G.nodes)

# to print edge labels
print(G.edges(data=True))

view_graph(getdotfile(G))



def mapping_conf(x):
	return x + ", hi"



def createGraphForConfidence(G, confidence):
	G_hi = G.copy()
	G_hi = nx.relabel_nodes(G_hi, lambda x : x + ", " + confidence)

	print (G_hi.nodes)
	# print (G_hi.edges(data = True))

	# print(G.edges.data("label"))

	edges = list(G_hi.edges(data=True))
	print ("here")
	for i in range(len(edges)):
		try : 
			edges[i][2]['label'] = edges[i][2]['label'] + ", " + confidence
		except : 
			# print (edges[i])
			pass
			# this is the init edge


	G_hi.remove_edges_from(list(G_hi.edges()))
	G_hi.add_edges_from(edges)

	return G_hi


def generate_hi_med_low_graphs(G):
	G_hi = createGraphForConfidence(G, "h")
	G_med = createGraphForConfidence(G, "m")
	G_low = createGraphForConfidence(G, "l")

	return G_hi, G_med, G_low



def get_label(x):
	print (x)
	return x.split(", ")[1]

def draw_edge_between_graphs(G):
	# input is the merged graph with low med hi nodes. 

	nodes = G.nodes(data=True)

	newedges = ["h", "m", "l"]


	new_edges_to_add = []

	for eachnode in nodes : 
		thisedges = G.edges(eachnode[0], data=True)
		# get the current label.
		cf = get_label(eachnode[0])




		for i in newedges : 
			if(i == cf) : 
				continue

			for eachedge in thisedges : 
				nedge = []

				nedge.append(eachedge[0])
				nedge.append(eachedge[1].split(", ")[0] + ", " + i)

				try : 
					label = eachedge[2]["label"].split(", ")[0] + ", " + i
					print (label)
					# nedge.append({ "label" : label})
				except : 
					print ("FAIL", eachedge)

				new_edges_to_add.append(nedge)


	G.add_edges_from(new_edges_to_add)

	return G 




def merge_hi_med_low_graphs(G_hi, G_med, G_low):

	G = nx.MultiDiGraph()

	G.add_nodes_from(G_hi.nodes(data=True))
	G.add_nodes_from(G_med.nodes(data=True))
	G.add_nodes_from(G_low.nodes(data=True))

	G.add_edges_from(G_hi.edges(data=True))
	G.add_edges_from(G_med.edges(data=True))
	G.add_edges_from(G_low.edges(data=True))


	G = draw_edge_between_graphs(G)


	view_graph(getdotfile(G))




def addHighMedLowStates(G):
	# for every node we create 3 new nodes. 
	# like 3 new graphs. as G_hi G_low G_med
	# between these graphs 
	# there is an edge between node for all x_low there should be an edge back to x_high
	# for all x_med there should be an edge back to x_high

	G_hi, G_med, G_low = generate_hi_med_low_graphs(G)
	merge_hi_med_low_graphs(G_hi, G_med, G_low)







r = reward(s,q)


def reward(s,q):
	# use q and s,
	r = reward_from_env(s)
	q = get_q_State_from_image(s)
	r2 = reward_from_dfa(q)
	return f(r,r2)



	# view_graph(getdotfile(G_low))


# addHighMedLowStates(G)

# use networkx to convert dfa dot to networkx graph so that it can be later modified. 




