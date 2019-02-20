import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self,cfg = None):
        super(TestModel,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(1,8,3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8,16,5,2,0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.fc = nn.Linear(29*16,5)

    def forward(self,x):
        out = self.layer(x)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out

