import torch
import torch.nn as nn
import numpy as np
from utils.config import cls_test

from models.testmodel import TestModel2 ,TestModel1
from models.resnet import ResNet18

model_map = {'TestModel1' : TestModel1,
             'TestModel2' : TestModel2,
             'ResNet18' : ResNet18,
            }
def create_model(cfg = None):
    base =  model_map[cfg['base_model']]
    model = base(cfg)
    return model

def test():
    cfg = cls_test
    model =  create_model(cfg)
    print(model)

