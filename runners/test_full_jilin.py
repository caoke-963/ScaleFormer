#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
@LastEditTime: 2020-06-23 17:50:08
@Description: test.py
'''

import os

from _bootstrap import add_project_root, resolve_from_root

add_project_root()
from utils.config  import get_config
from solver.testsolver_full import Testsolver
# from solver.testsolver_fullpatch import Testsolver
# 仅设置一块可见
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    cfg = get_config(resolve_from_root('options/option_full_jilin.yml'))
    solver = Testsolver(cfg)
    solver.run()
    
