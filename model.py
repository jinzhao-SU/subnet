import torch
import torch.nn as nn
import torchvision


class PreTrain(nn.Module):
    def __init__(self):
        super(PreTrain, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(10000, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 4096)
        self.fc4 = nn.Linear(4096,10000)

        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(4096)

    def init_parameters(self):
        torch.nn.init.zeros_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc4.weight)

    def _subnet_forward(self, x):
        x = self.fc1(x)
        # x = torch.squeeze(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = x.view(-1, 100, 100)
        x = torch.sigmoid(x)
        return x

    def forward(self, x_image=None):
        x_image = x_image.float()
        #print("input shape", x_image.shape)
        shape = x_image.shape
        # print("x shape", x_image.shape)
        # x_image = x_image.permute(0,)
        x_image = torch.squeeze(x_image)
        x = self._subnet_forward(x_image)
        #print("xoutput", x.shape) #32 100 100
        return x

# if __name__ == '__main__':
