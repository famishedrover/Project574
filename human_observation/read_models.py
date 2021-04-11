import os


import classifiers.misc as cMisc
import human_observation.config as config 






import torch
from classifiers.network import Net 
import torch.optim as optim
from torchvision import transforms
import PIL.Image

import classifiers.config as cConfig



image_transforms = 	transforms.Compose([
					transforms.Resize(cConfig.TRAIN_IMAGE_SHAPE),
					transforms.ToTensor(),
					])


def load_model(fluent):
	tmp = os.path.join(config.SAVED_MODELS_PATH, fluent)
	MODEL_PATH = os.path.join(tmp, config.MODEL_NAME)


	model = Net()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	checkpoint = torch.load(MODEL_PATH)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	model.eval()

	return model



def load_all_models():
	models_names = cMisc.import_fluents(config.SAVED_MODELS_PATH)
	print ("Loading Models for ", models_names)

	fluent_classifier = {}

	for model_name in models_names : 
		model = load_model(model_name)
		fluent_classifier[model_name]  = model

	return fluent_classifier



def transform_image(image):
	image = image_transforms(image)
	return image
	


print(load_all_models())


exit()


GET_SOFTMAX = torch.nn.Softmax(dim=1)




# TEST MODEL ON AN IMAGE : 



def read_image_for_network(path):
	rgba_image = PIL.Image.open(path)
	image  = rgba_image.convert('RGB')

	image = image_transforms(image)

	return image


# image = image_transforms(env.render(image=True))


# two things 
# first - > wrap classifier with dfa, reward


IMAGE_PATH = "./fluents/is_on_edge_clear/positive/1.png"

tensor_image = read_image_for_network(IMAGE_PATH)
tensor_image = tensor_image.unsqueeze(0)

print (tensor_image.shape)



result = model(tensor_image)
probs = GET_SOFTMAX(result)
print(result)
print(probs)

