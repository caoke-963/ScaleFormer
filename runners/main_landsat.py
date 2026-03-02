#!/usr/bin/env python
# coding=utf-8

import argparse
import os

from _bootstrap import add_project_root, resolve_from_root

add_project_root()
from utils.config import get_config
from solver.unisolver_patcheval import Solver
# from solver.gf_solver import Solver
# from solver.midnsolver import Solver
# from solver.innformersolver import Solver

# 仅设置一块可见
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 设置多块可见
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N_SR')
    parser.add_argument('--option_path', type=str, default='options/option_landsat.yml')
    opt = parser.parse_args()
    cfg = get_config(resolve_from_root(opt.option_path))
    solver = Solver(cfg)
    solver.run()
    
