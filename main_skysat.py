#!/usr/bin/env python
# coding=utf-8


from utils.config import get_config
from solver.unisolver_patcheval import Solver
# from solver.gf_solver import Solver
# from solver.midnsolver import Solver
# from solver.innformersolver import Solver
import argparse

import os
# 仅设置一块可见
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# 设置多块可见
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N_SR')
    parser.add_argument('--option_path', type=str, default='option_skysat.yml')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)
    solver = Solver(cfg)
    solver.run()
    