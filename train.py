import argparse
import torch
import torch.nn as nn
import os
from models.testmodel import TestModel
from models.bulid_model import create_model
from utils.MyDataset import MyDataset ,MyDataset1

from shutil import copyfile



def train():
    model_name = 'TestModel'

    '''cfg load'''

    '''parameters which will be optimized  '''

    trainset_root = '/train/results/ecg_cls/'
    train_images_set = (('list_index','all_list.txt'),('list_index','new_list.txt'))
    num_epoch = 44
    is_pretrained = True
    load_model_path = './weight/f1.pth'
    save_model_dir = './weight'
    use_gpu = True
    gpu_id = 1

    '''there is a problem about execute speed how to contrast device. and need to be optim'''
    #device = 'cpu' if use_gpu else 'cuda:{}'.format(gpu_id)
    device = 'cuda:1'
    
    
    '''load the train model'''
    model = create_model(model_name)
   
    ''' load pretrained model'''
    if(is_pretrained):
        model.load_state_dict(torch.load(load_model_path))
        print('load pretrained end')
        
    ''' dataset load'''     
    train_list_path = './all_list.txt'
    #train_set = MyDataset(train_list_path)
    train_set = MyDataset1(trainset_root,train_images_set)
    train_loader = torch.utils.data.DataLoader(dataset = train_set , batch_size = 10 , shuffle = True)
    print(len(train_loader))


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
    else:
        print('file {} is not exist'.format(file_path))
if __name__ == '__main__':
    train()
