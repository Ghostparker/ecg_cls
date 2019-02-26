import torch
import torch.nn as nn

class TestModel2(nn.Module):
    def __init__(self,cfg = None):
        super(TestModel2,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(1,8,3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8,16,5,2,0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.fc = nn.Linear(29*16 , 5)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self,x ,mode = 'train' ):
        out = self.layer(x)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        if(mode == 'eval'):
            out = self.softmax(out)
        return out

class testModel(nn.Module):
    def __init__(self,cfg = None,classifiers= None):
        super(testModel,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(1,8,3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8,16,5,2,0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.classifiers = nn.ModuleList(classifiers)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self,x ,mode = 'train' ):
        out = self.layer(x)
        out = out.view(out.size(0),-1)
        outline = list()
        for k ,v in enumerate(self.classifiers):
            out = v(x)
            if(mode == 'eval'):
                outline.append(self.softmax(out))
            else:
                outline.append(out)
        return outline
        
def  add_classifiers( in_channels  = 464, num_classes_list = [5,]):
    layers = []
    for num_class in num_classes_list:
        layers +=[nn.Linear(in_channels , num_class)]
    return layers

def TestModel1(num_classes = [5,]):
    num = 29 * 16
    classifiers = add_classifiers(num , num_classes)
    return testModel(cfg = None , classifiers = classifiers)

