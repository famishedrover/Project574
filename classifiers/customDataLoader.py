from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import random
import PIL.Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode



class FluentsDataset(Dataset):

	def __init__(self, root_dir, concept_name, pos_count=10, neg_count=10, exact=False, transform=None):
		# selects upto pos_count and neg_count samples. 
		# if exact=True then exactly selects the number of samples otherwise returns an error. 
	  
		self.root_dir = root_dir
		self.transform = transform
		self.pos_count = pos_count
		self.neg_count = neg_count
		self.concept_name = concept_name
		self.exact = exact
		self.transform = transform

		self.positive_paths, self.negative_paths = self.select_batch()

		self.allPaths = self.positive_paths + self.negative_paths
		random.shuffle(self.allPaths)

		print (self.allPaths)

	def select_batch(self):
		POSITIVE_PATH = os.path.join(self.root_dir, self.concept_name, "positive")
		NEGATIVE_PATH = os.path.join(self.root_dir, self.concept_name, "negative")

		positive_paths = [os.path.join(POSITIVE_PATH, x) for x in os.listdir(POSITIVE_PATH)]
		negative_paths = [os.path.join(NEGATIVE_PATH, x) for x in os.listdir(NEGATIVE_PATH)]

		if(not self.exact) : 
			self.pos_count = min(len(positive_paths), self.pos_count)
			self.neg_count = min(len(negative_paths), self.neg_count)


		try : 
			positive_paths = random.sample(positive_paths, self.pos_count)
			negative_paths = random.sample(negative_paths, self.neg_count)
		except ValueError: 
			print ("Sample Count expected is not present in the directory! Add samples or choose exact=False")
			exit()

		return positive_paths, negative_paths



	def __len__(self):
		return len(self.allPaths)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = self.allPaths[idx]

		rgba_image = PIL.Image.open(img_name)
		image  = rgba_image.convert('RGB')

		thisclass = 1

		if ("negative" in img_name) : 
			thisclass = 0

		

		if self.transform:
			image = self.transform(image)




		sample = {'image': image, 'class': thisclass}

		return sample



def view_tensor_image(image):
	x = transforms.ToPILImage()(image)
	x.show()



def get_test_train_split(full_dataset, percent=0.8):
	train_size = int(percent * len(full_dataset))
	test_size = len(full_dataset) - train_size
	train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

	return train_dataset, test_dataset

if __name__ == "__main__":
	onTop_dataset = FluentsDataset("./fluents", "onTop", pos_count=2, neg_count=1, exact=False, transform = transforms.Compose([
																											transforms.Resize([720,720]),
																											transforms.ToTensor(),
																											 ])  )

	train_dataset, test_dataset = get_test_train_split(onTop_dataset, 0.8)

	train_data_loader = torch.utils.data.DataLoader(train_dataset,
										  batch_size=1,
										  shuffle=True,
										  num_workers=1)

	test_data_loader = torch.utils.data.DataLoader(test_dataset,
										  batch_size=1,
										  shuffle=True,
										  num_workers=1)


	print (len(train_data_loader))
	print (len(test_data_loader))

	# for x in train_data_loader : 
	# 	images = x["image"]
	# 	labels = x["class"]

	# 	image = images[0]
	# 	tensor_image = image


		# plt.imshow(tensor_image.permute(1,2,0))
		# plt.show()
		# plt.pause(5)
		
		

		
