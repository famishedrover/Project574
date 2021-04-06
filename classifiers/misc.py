import os 
import config

def import_fluents(root):
	fluents = os.listdir(root)
	for x in fluents :
		if("." in x):
			fluents.remove(x)

	return fluents



def create_dir_ifn(path):
	if not os.path.exists(path):
		os.mkdir(path)
		print ("Created Dir ", path)

def create_fluents_model_dir(root, fluents):
	ROOT_PATH_FOR_MODELS = os.path.join(root, config.FLUENTS_MODEL_DIR_NAME)

	create_dir_ifn(ROOT_PATH_FOR_MODELS)

	paths = {}

	for fluent in fluents : 
		FLUENT_PATH = os.path.join(ROOT_PATH_FOR_MODELS, fluent)
		create_dir_ifn(FLUENT_PATH)

		paths[fluent] = FLUENT_PATH

	return paths





