import torch
import torchvision as tv 

from torchvision import transforms

train_data = tv.datasets.ImageFolder('./fluents/onTop', transform=transforms.Compose([
																	transforms.ToTensor(),
																	transforms.Resize((84,84))]))

print (type(train_data))

data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=2,
                                          shuffle=True,
                                          num_workers=4)


print (data_loader)

for x in enumerate(data_loader) : 
	print (x)

