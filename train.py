import argparse
import torch
import torch.nn as nn
import os
from models.testmodel import TestModel2 as TestModel
from models.bulid_model import create_model
from utils.MyDataset import MyDataset ,MyDataset1
import time
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
    iteration = 0 
    max_iter, save_iter ,log_iter = 400  ,100,20
    device = 'cuda:{}'.format(gpu_id) if use_gpu else 'cpu'
    #device = 'cuda:1'
    
    
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


    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters() , lr = 1e-3)
    
    t1 = time.time()

    for epoch in range(num_epoch):
        for idx , (images , labels) in enumerate(train_loader):
            

            model.train()
            images = images.float().to(device)
            labels = labels.long().to(device)
            outputs = model(images)
        
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(iteration == 0 or (iteration % log_iter == 0)):
                print('iteration  {} loss {:.4f}'.format( iteration , loss.item()))

            if( iteration == 0 or ( iteration != 0 and iteration % save_iter == 0)):
                save_checkpoint(model ,  save_model_dir , 'model_{}.pth'.format(iteration))
            iteration += 1
            if(iteration >= max_iter):
                raise SystemExit('train is done')
    t2= time.time()
    print(t2-t1)

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
        print('file {} is not exist'.format(path))
if __name__ == '__main__':
    train()
