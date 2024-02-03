import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

# The Neural Network structure
class Model(nn.Module):
    def __init__(self, input_layer = 50, h1 = 100, h2 = 300, h3 = 50, output_layer = 100):
        super().__init__()
        self.layer1 = nn.Linear(input_layer, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, h3)
        self.output = nn.Linear(h3, output_layer)

    #Connects all the layers of the NN
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.output(x)
        return x