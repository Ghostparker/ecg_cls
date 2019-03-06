import argparse
import torch
import torch.nn as nn
import os
from models.bulid_model import create_model
from utils.MyDataset import MyDataset ,MyDataset2
from utils.config import cls_test
import time

from shutil import copyfile



def train():

    '''cfg load'''

    '''parameters which will be optimized  '''

    cfg = cls_test
    batch_size = 100
    is_pretrained = True
    load_model_path = './weight/model_40000.pth'
    save_path = './results/'
    error_path = './results/error.txt'
    device = 'cuda:{}'.format(cfg['gpu_id']) if  cfg['use_gpu'] else 'cpu'
    num_classes = [5,]
    #device = 'cuda:1'
    
    
    '''load the train model'''
    print('Model {}'.format(cfg['base_model']))
    print(device)
    model = create_model(cfg)
   
    ''' load pretrained model'''
    model.load_state_dict(torch.load(cfg['eval_model_path']))
        
    ''' dataset load'''     
    #train_set = MyDataset(train_list_path)
    test_set = MyDataset2(cfg['testset_root'],cfg['test_images_set'])
    train_loader = torch.utils.data.DataLoader(dataset = test_set , 
                                        batch_size = cfg['test_batch_size'] , shuffle = False)


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
