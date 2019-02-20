import torch
import torch.nn as nn
from models.testmodel import TestModel
from models.bulid_model import create_model
from utils.MyDataset import MyDataset
model_name = 'TestModel'
#net = TestModel

#######################
'''cfg load'''


num_epoch = 50 
device = 'cuda:1'

model = create_model(model_name)

#from torch.utils import data.DataLoader
##################################
### train_loader 
train_list_path = './all_list.txt'
train_set = MyDataset(train_list_path)
train_loader = torch.utils.data.DataLoader(dataset = train_set , batch_size = 10 , shuffle = False)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters() , lr = 1e-3)
for epoch in range(num_epoch):
    for idx , (images , labels) in enumerate(train_loader):
        images = images.float().to(device)
        labels = labels.long().to(device)
        outputs = model(images)
    
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(idx == 1):
            print('epoch {} batch {}loss {}'.format(epoch , idx , loss.item()))
