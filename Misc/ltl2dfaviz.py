from ltlf2dfa.parser.pltlf import PLTLfParser
from graphviz import Source


FORMULA = "H(a -> Y b)"



parser = PLTLfParser()
formula_str = FORMULA
formula = parser(formula_str)       # returns a PLTLfFormula

print(formula)                      # prints "H(a -> Y (b))"

dfa = formula.to_dfa()

# dfa.replace("\n", "")





s = Source(dfa, filename="dfa.gv", format="png")
s.view()



# use networkx to convert dfa dot to networkx graph so that it can be later modified. 