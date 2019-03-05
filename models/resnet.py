import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels , out_channels , stride = 1  , downsample = None):
        super(ResidualBlock,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels , out_channels , 3 , stride , 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels , out_channels , 3 , 1 , 1),
            nn.BatchNorm1d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self,x ):
        residual = x
        out = self.layer(x)
        if(self.downsample):
            residual = self.downsample(x)
        out += residual
        
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self , block , layers, num_classes = 5):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.prelayer = nn.Sequential(
            nn.Conv1d(1,self.in_channels,3,1,1),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(block , 16 ,layers[0],stride = 1)
        self.layer2 = self.make_layer(block , 32 ,layers[1],stride = 2)
        self.layer3 = self.make_layer(block , 128,layers[2],stride = 2)
        #self.layer4 = self.make_layer(block , 256,layers[3],stride = 2)
        self.fc = nn.Linear(256,num_classes)
        self.softmax = nn.Softmax(dim = -1)

    def make_layer(self , block , out_channels , blocks , stride = 1):
        downsample = None
        if(stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels,3,stride , 1),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels , out_channels , stride , downsample))
        self.in_channels = out_channels
        for i in range(1,blocks):
            layers.append(block(out_channels , out_channels))
        return nn.Sequential(*layers)
    def forward(self,x , mode = 'train'):
        out = self.prelayer(x) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool1d(out,7)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        if(mode == 'eval'):
            out = self.softmax(out)
        return out



