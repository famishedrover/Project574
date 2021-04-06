import torch
from network import Net 
import torch.optim as optim
from torchvision import transforms
import PIL.Image

import config

MODEL_PATH = './fluents_models/is_on_edge_clear/model_150_150_False.pt'

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']


model.eval()
# - or -
# model.train()


GET_SOFTMAX = torch.nn.Softmax(dim=1)




# TEST MODEL ON AN IMAGE : 

image_transforms = 	transforms.Compose([
					transforms.Resize(config.TRAIN_IMAGE_SHAPE),
					transforms.ToTensor(),
					])

def read_image_for_network(path):
	rgba_image = PIL.Image.open(path)
	image  = rgba_image.convert('RGB')

	image = image_transforms(image)

	return image




IMAGE_PATH = "./fluents/is_on_edge_clear/positive/1.png"

tensor_image = read_image_for_network(IMAGE_PATH)
tensor_image = tensor_image.unsqueeze(0)

print (tensor_image.shape)



result = model(tensor_image)
probs = GET_SOFTMAX(result)
print(result)
print(probs)

