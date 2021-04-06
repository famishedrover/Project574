import networkx as nx
import pprint



class NeverClaim:
	def __init__(self, NEVER_CLAIM_PATH = "never_claim.txt") : 
		self.NEVER_CLAIM_PATH = NEVER_CLAIM_PATH

		self.rawFile = self.read_raw_file()
		self.SymbolTable = set()
		self.cleanData = self.convert_raw2cleanData()
		self.G = None
		self.nodes_from_raw = None

		self.Nodes2NameTable = {}
		self.Name2NodesTable = {}

	def read_raw_file(self):
		with open (self.NEVER_CLAIM_PATH) as f : 
			x = f.read()
		return x

	def convert_raw2cleanData(self):

		x = self.rawFile
		x = x.split("\n")
		x = x[1:-1]

		clean_data = []
		nodes = []

		for t in x :
			if("::" in t) :
				t = t.replace(":: ", "")
				t = t.split(" -> ")
				t = [x.strip() for x in t]
				assert len(t) == 2, print ("Length of Edge not 2")

				# t[0] = t[0][1:-1] # remove paranthesis
				t[0] = self.fix_edge(t[0])
				self.populate_symbol_table(t[0])
				t[1] = t[1].replace("goto ","")


				# print ("Edge", t)

			elif (":" in t): 
				t = t.replace(":", "").strip()
				nodes.append(t)
				# print ("Node", t)
			else : 
				continue

			clean_data.append(t)

		self.nodes_from_raw = nodes

		return clean_data



	def fix_edge(self, x):
		x = x.replace("!", "not ")
		x = x.replace("&&", " and ")
		x = x.replace("||", " or ")
		return x

	def populate_symbol_table(self, x):
		# x is edge text
		# s is symbol table instance.

		replacewords = ["(", ")", "not", "and", "or"]

		for i in replacewords : 
			x = x.replace(i, " ")

		x = x.split(" ")

		x = [t.strip() for t in x]

		newx = []
		for t in x : 
			if(len(t) > 1):
				newx.append(t)

		x = newx 

		for t in x : 
			self.SymbolTable.add(t)




	def getnxGraph(self):

		G = nx.MultiDiGraph()

		x = self.cleanData 

		assert type(x[0]) == type(''), "Check Clean Data File. First Line must be a node"

		prev_node = None 
		for t in x : 
			if(type(t) == type([])):
				# Edge 
				data = t[0]
				to_node = t[1]

				G.add_edge(prev_node, to_node, data=data)

			else : 
				# Node 
				prev_node = t


		self.G = G 


		for node in G.nodes : 
			c = 'd'
			# default node. 

			if('init' in node) : 
				c = 'i'
			elif('accept' in node) : 
				c = 'a'
			else : 
				c = 'd'

			G.nodes[node]['type'] = c


		# relabel nodes. 
		all_nodes = list(G.nodes)
		for i in range(len(all_nodes)) : 
			self.Name2NodesTable[all_nodes[i]] = i
			self.Nodes2NameTable[i] = all_nodes[i]

		G = nx.relabel_nodes(G, self.Name2NodesTable)

		return G






# exec(foo + " = 'something else'")






if __name__ == "__main__" :
	# populate_symbol_table(SymbolTable, "(not(on_edge))")

	ncm = NeverClaim("never_claim.txt")

	print ("Symbol Table : ", ncm.SymbolTable)

	print ("clean_data")

	pprint.pprint(ncm.cleanData)


	print("Graph")
	G = ncm.getnxGraph()
	print (G.nodes())
	print (G.edges())
