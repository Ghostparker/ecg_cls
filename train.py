import argparse
import torch
import torch.nn as nn
import os
from models.bulid_model import create_model
from utils.MyDataset import MyDataset ,MyDataset1,MyDataset2

import time
from utils.config import cls_test
from shutil import copyfile



def train():
    model_name = 'TestModel2'

    '''cfg load'''

    '''parameters which will be optimized  '''
    cfg = cls_test
    # trainset_root = 'D:/python program/ecg_cls-master/'
    # train_images_set = (('index_dataset','trainlist.txt'),)

    # batch_size = 100
    # is_pretrained = False
    load_model_path = './weight/f1.pth'
    save_model_dir = './weight'

    gpu_id = cfg['gpu_id']
    iteration = 0 
    max_iter, save_iter ,log_iter = cfg['max_iter'],cfg['save_iter'],cfg['log_iter']
    num_epoch = max_iter
    device = 'cuda:{}'.format(cfg['gpu_id']) if cfg['use_gpu'] else 'cpu'
    print(device)
    #device = 'cuda:1'
    
    
    '''load the train model'''
    print('Model {}'.format(cfg['base_model']))
    model = create_model(cfg)
   
    ''' load pretrained model'''
    if(cfg['is_pretrained']):
        model.load_state_dict(torch.load(load_model_path))
        print('load pretrained end')
    else:
        print('train from beginning')

    ''' dataset load'''     
    train_list_path = './all_list.txt'
    #train_set = MyDataset(train_list_path)
    train_set = MyDataset1(cfg['trainset_root'],cfg['train_images_set'])
    train_loader = torch.utils.data.DataLoader(dataset = train_set , batch_size = cfg['train_batch_size'] , shuffle = True)
    print('trainloader end')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    now_lr = cfg['learningrate']
    optimizer = torch.optim.SGD(model.parameters() , lr = now_lr)
    if(cfg['lr_change'] == 'lr_steps'):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer , step_size = cfg['step_size'] , gamma = cfg['gamma'])
    elif(cfg['lr_change'] == 'cosine'):
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max_iter) 
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
                print('iteration {} lr {}  loss {:.4f}'.format( iteration , optimizer.param_groups[0]['lr'] ,loss.item()))

            if( iteration == 0 or ( iteration != 0 and iteration % save_iter == 0)):
                save_checkpoint(model ,  save_model_dir , '{}-{}.pth'.format(cfg['base_model'],iteration))
            
            if( cfg['lr_change'] == 'lr_steps'):
                if(iteration in cfg['lr_steps']):
                    #scheduler.step()
                    print('i am uncle t')
                    print('tsl')
                    now_lr *= cfg['gamma']
                    adjust_learning_rate(optimizer , now_lr)
            elif( cfg['lr_change'] == 'cosine'):
                scheduler.step()
            else:
                pass
            iteration += 1
            if(iteration >= max_iter):
                save_checkpoint(model ,  save_model_dir , 'model_{}.pth'.format(iteration))
                t2= time.time()
                print(t2-t1)
                raise SystemExit('train is done')

''' adjust the learning in trianing '''
def adjust_learning_rate(optimizer , lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
