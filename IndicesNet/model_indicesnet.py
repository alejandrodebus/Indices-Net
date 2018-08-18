import torch
import torch.nn as nn
import torch.nn.functional as F

class IndicesNet(nn.Module):

    def __init__(self, model_dcae):
        super(IndicesNet, self).__init__()

        self.dcae = model_dcae

        self.conv_reg1 = nn.Conv2d(16, 16, kernel_size = 5, stride = 1, padding = 2)
        self.conv_reg2 = nn.Conv2d(16, 16, kernel_size = 5, stride = 1, padding = 2)
        self.conv_reg3 = nn.Conv2d(16, 8, kernel_size = 80, stride = 1, padding = 0)

    def forward(self, x):

        _, x = self.dcae(x)
        x = self.conv_reg1(x)
        x = self.conv_reg2(x)
        x = self.conv_reg3(x)

        x = x.view(x.size(0), -1)

        return x
