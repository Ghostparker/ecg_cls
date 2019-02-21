import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as  np
class MyDataset(Dataset):
    def __init__(self,root , transform = None , target_transform = transforms.ToTensor() , 
                channels = 1):
        self.root= root
        self.channels= channels
        self.ids = list()
        fp = open(root)
        for line in fp:
            words = line.strip().split(' ')
            self.ids.append({
                'pic_path' : words[0],
                'label' : int(words[1]),
            })

    def __getitem__(self,index):
        return self._pull_item(index , self.channels)

    def __len__(self):
        return len(self.ids)

    def _pull_item(self, index , channels,maxlen = 100,end_point = 64):
        fp = open(self.ids[index]['pic_path'])
        tlabel = self.ids[index]['label']
        new_src = np.zeros([channels , maxlen],dtype=np.float32)
        for idx ,line in enumerate(fp):
            words = line.strip().split(' ')
            for id_sig , one_single in enumerate(words):
                new_src[idx][id_sig] = float(one_single)

        return torch.from_numpy(new_src[:,:end_point]) , tlabel


class MyDataset1(Dataset):
    def __init__(self, root_dir , images_set = [('',''),], transform = None ,
                target_transform = transforms.ToTensor() , 
                channels = 1):
        self.root= root_dir
        self.images_set = images_set 
        self.channels= channels
        self.ids = list()
        for data_name , filename in self.images_set:
            testfile = os.path.join(self.root , data_name , filename)
            
            fp = open(testfile)
            for line in fp:
                words = line.strip().split(' ')
                self.ids.append({
                    'pic_path' : os.path.join(self.root,words[0]),
                    'label' : int(words[1]),
                })
            fp.close()

    def __getitem__(self,index):
        return self._pull_item(index , self.channels)

    def __len__(self):
        return len(self.ids)

    def _pull_item(self, index , channels,maxlen = 100,end_point = 64):
        fp = open(self.ids[index]['pic_path'])
        tlabel = self.ids[index]['label']
        new_src = np.zeros([channels , maxlen],dtype=np.float32)
        for idx ,line in enumerate(fp):
            words = line.strip().split(' ')
            for id_sig , one_single in enumerate(words):
                new_src[idx][id_sig] = float(one_single)

        return torch.from_numpy(new_src[:,:end_point]) , tlabel

