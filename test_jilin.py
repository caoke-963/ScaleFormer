#!/usr/bin/env python
# coding=utf-8

from utils.config  import get_config
from solver.testsolver import Testsolver

# from solver.midntestsolver import Testsolver
# from solver.inntestsolver import Testsolver
import os
# 仅设置一块可见
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    cfg = get_config('option_jilin.yml')
    solver = Testsolver(cfg)
    solver.run()
    