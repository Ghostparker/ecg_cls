import argparse
import torch
import torch.nn as nn
import os
from models.bulid_model import create_model
from utils.MyDataset import MyDataset ,MyDataset2
import time
from shutil import copyfile



def train():
    model_name = 'TestModel2'

    '''cfg load'''

    '''parameters which will be optimized  '''

    trainset_root = '/train/results/candelete/t2/ecg_cls/'
    train_images_set = (('index_dataset','testlist.txt'),)
    batch_size = 100
    is_pretrained = True
    load_model_path = './weight/model_40000.pth'
    save_path = './results/'
    error_path = './results/error.txt'
    use_gpu = True
    gpu_id = 0
    device = 'cuda:{}'.format(gpu_id) if use_gpu else 'cpu'
    num_classes = [5,]
    #device = 'cuda:1'
    
    
    '''load the train model'''
    model = create_model(model_name)
   
    ''' load pretrained model'''
    model.load_state_dict(torch.load(load_model_path))
        
    ''' dataset load'''     
    train_list_path = './all_list.txt'
    #train_set = MyDataset(train_list_path)
    train_set = MyDataset2(trainset_root,train_images_set)
    train_loader = torch.utils.data.DataLoader(dataset = train_set , batch_size = batch_size , shuffle = False)


    model = model.to(device)
    model.eval() 
    t1 = time.time()

    for idx , (images , labels, path) in enumerate(train_loader):
        model.eval()
        images = images.float().to(device)
        labels = labels.long().to(device)
        outputs = model(images,'eval')
        
        save_record(os.path.join(save_path,'result_{}.txt'.format(0)) ,error_path,path , labels , outputs)
    
    t2= time.time()
    print(t2-t1)


def save_record(spath ,error_path,fpath , groundtruth , confs):
    _ , predict = torch.max(confs.data,1)
    fp1 = open(error_path , 'a')
    fp = open(spath , 'a')
    for idx , path in enumerate(fpath):
        if(groundtruth[idx].data != predict[idx].data):
            fp1.write('{} {} {}\n'.format(path,groundtruth[idx] ,predict[idx].data))
        fp.write('{} {} {}'.format(path,groundtruth[idx] ,predict[idx].data))
        for conf in confs[idx]:
            fp.write(' {:.4f}'.format(conf.data.cpu().numpy()))
        fp.write('\n')
    fp.close()
    fp1.close()
if __name__ == '__main__':
    train()
