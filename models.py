import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(5000*3, 2500)
        self.l2 = nn.Linear(2500, 4000)
        self.l3 = nn.Linear(4000, 2562*3)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x).reshape((-1, 2562, 3))
        
        return x
