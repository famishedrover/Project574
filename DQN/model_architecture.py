import torch 
import torch.nn as nn
import torch.nn.functional as F



class QNetwork(nn.Module):

    def __init__(self, img_size, outputs, seed, dfa_one_hot_size):
        super(QNetwork, self).__init__()

        h,w = img_size
        self.dfa_embedding_size = dfa_one_hot_size

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size + self.dfa_embedding_size, 32)
        self.head2 = nn.Linear(32, 16)
        self.head3 = nn.Linear(16, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, q):

        # print ("X SHAPE = ", x.shape)

        x = x.permute(0,3,1,2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        # x = torch.hstack([x,q])
        x = torch.cat([x,q], dim=1)

        x = self.head(x)
        x = self.head2(x)
        x = self.head3(x)
        return x


if __name__ == "__main__" : 
    x = torch.zeros((1,64,64,3))
    # x = x.permute(0,3,1,2)
    model = QNetwork((64,64), 7, 0, 6)

    q = torch.zeros(1,6)

    print (model(x,q))