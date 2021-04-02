from customDataLoader import FluentsDataset, get_test_train_split
import os 
import misc

from torchvision import transforms

from network import Net 
import torch
import torch.nn as nn

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn




FLUENTS_ROOT_DIR = "./fluents"

fluents = misc.import_fluents(FLUENTS_ROOT_DIR)

print (fluents)



import torch.optim as optim





for eachFluent in fluents : 

	print ("Current Fluent : ", eachFluent)

	fluent_dataset = FluentsDataset(FLUENTS_ROOT_DIR, eachFluent, pos_count=2, neg_count=2, exact=False, transform = transforms.Compose([
																											transforms.Resize([84,84]),
																											transforms.ToTensor(),
																											 ])  )

	train_dataset, test_dataset = get_test_train_split(fluent_dataset, 0.8)

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


	for epoch in range(2):  # loop over the dataset multiple times

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
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
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



