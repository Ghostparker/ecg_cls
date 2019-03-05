import torch
import torch.nn as nn
import numpy as np

from models.testmodel import TestModel2 ,TestModel1

model_map = {'TestModel1' : TestModel1,
             'TestModel2' : TestModel2,
            }
def create_model(cfg = None):
    base =  model_map[cfg['base_model']]
    model = base(cfg)
    return model


