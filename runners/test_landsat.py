#!/usr/bin/env python
# coding=utf-8

import os

from _bootstrap import add_project_root, resolve_from_root

add_project_root()
from utils.config  import get_config
from solver.testsolver import Testsolver
# from solver.midntestsolver import Testsolver
# from solver.inntestsolver import Testsolver
# 仅设置一块可见
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if __name__ == '__main__':
    cfg = get_config(resolve_from_root('options/option_landsat.yml'))
    solver = Testsolver(cfg)
    solver.run()
    
