import argparse
import torch
import torch.nn as nn
import os
from models.testmodel import TestModel
from models.bulid_model import create_model
from utils.MyDataset import MyDataset
from shutil import copyfile



def train():
    model_name = 'TestModel'

    '''cfg load'''

    '''parameters which will be optimized  '''
    num_epoch = 44
    device = 'cuda:1'
    is_pretrained = True
    load_model_path = './weight/f1.pth'
    save_model_dir = './weight'
    
    '''load the train model'''
    model = create_model(model_name)
   
    ''' load pretrained model'''
    if(is_pretrained):
        model.load_state_dict(torch.load(load_model_path))
        print('load pretrained end')
        
     
    train_list_path = './all_list.txt'
    train_set = MyDataset(train_list_path)
    train_loader = torch.utils.data.DataLoader(dataset = train_set , batch_size = 10 , shuffle = True)

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
    save_checkpoint(model , save_model_dir , 'f2.pth')
''' adjust the learning in trianing '''
def adjust_learning_rate(optimizer , lr):
    pass

'''save model '''
def save_checkpoint(model , path , name):
    final_path = os.path.join(path , name)
    if(os.path.exists(path)):
        torch.save(model.state_dict(),final_path)
        print('{} save end'.format(final_path))
       
    pass 
if __name__ == '__main__':
    train()
