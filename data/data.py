#!/usr/bin/env python
# coding=utf-8

from os.path import join
from torchvision.transforms import Compose, ToTensor
from .dataset import Data, Data_test, Data_eval
from torchvision import transforms
import torch, numpy  #h5py, 
import torch.utils.data as data

def transform():
    return Compose([
        ToTensor(),
    ])
    
def get_data(cfg, mode):
    data_dir_ms = join(mode, cfg['source_ms'])
    data_dir_pan = join(mode, cfg['source_pan'])
    data_dir_mask = join(mode,"mask")
    cfg = cfg
    return Data(data_dir_ms, data_dir_pan, cfg, transform=transform(),data_dir_mask=data_dir_mask)
    
def get_test_data(cfg, mode):
    data_dir_ms = join(mode, cfg['test']['source_ms'])
    data_dir_pan = join(mode, cfg['test']['source_pan'])
    data_dir_mask = join(mode, "mask")
    cfg = cfg
    return Data_test(data_dir_ms, data_dir_pan, cfg, transform=transform(),data_dir_mask=data_dir_mask)

def get_eval_data(cfg, data_dir, upscale_factor):
    data_dir_ms = join(mode, cfg['test']['source_ms'])
    data_dir_pan = join(mode, cfg['test']['source_pan'])
    cfg = cfg
    return Data_eval(data_dir_ms, data_dir_pan, cfg, transform=transform())