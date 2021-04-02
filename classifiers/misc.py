import os 

def import_fluents(root):
	fluents = os.listdir(root)
	for x in fluents :
		if("." in x):
			fluents.remove(x)

	return fluents