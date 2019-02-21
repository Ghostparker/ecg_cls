import torch
import torch.nn as nn
import numpy as np

from models.testmodel import TestModel2 

model_map = {'TestModel' : TestModel2,
            }
def create_model(name = 'TestModel' , cfg = None):
    base =  model_map[name]
    model = base(cfg)
    return model
