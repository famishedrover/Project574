from customDataLoader import FluentsDataset, get_test_train_split
import os 
import misc

from torchvision import transforms

from network import Net 
import torch
import torch.nn as nn
import torch.optim as optim

import config

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn




FLUENTS_ROOT_DIR = config.FLUENTS_ROOT_DIR

fluents = misc.import_fluents(FLUENTS_ROOT_DIR)

fluent_model_paths = misc.create_fluents_model_dir(root="./", fluents=fluents)

print ("All Fluents ", fluents)








for eachFluent in fluents : 

	print ("Current Fluent : ", eachFluent)

	fluent_dataset = FluentsDataset(FLUENTS_ROOT_DIR, eachFluent, pos_count=config.POS_COUNT, neg_count=config.NEG_COUNT, exact=config.EXACT_NUM_SAMPLES, transform = transforms.Compose([
																											transforms.Resize(config.TRAIN_IMAGE_SHAPE),
																											transforms.ToTensor(),
																											 ])  )

	train_dataset, test_dataset = get_test_train_split(fluent_dataset, 0.7)

	train_data_loader = torch.utils.data.DataLoader(train_dataset,
										  batch_size=1,
										  shuffle=True,
										  num_workers=0)

	test_data_loader = torch.utils.data.DataLoader(test_dataset,
										  batch_size=1,
										  shuffle=True,
										  num_workers=0)

	net = Net()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


	for epoch in range(config.NUM_EPOCHS):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(train_data_loader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data["image"], data["class"]

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			# if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
			running_loss = 0.0

	print('Finished Training')




	correct = 0
	total = 0
	with torch.no_grad():
		for data in test_data_loader:
			inputs, labels = data["image"], data["class"]
			outputs = net(inputs)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the test images: %d %%' % (
		100 * correct / total))


	MODEL_NAME = 'model_' + str(config.POS_COUNT) + '_' + str(config.NEG_COUNT) + '_' + str(config.EXACT_NUM_SAMPLES) + '.pt'
	MODEL_PATH = os.path.join(fluent_model_paths[eachFluent], MODEL_NAME)
	torch.save({
            'epoch': config.NUM_EPOCHS,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_PATH)





