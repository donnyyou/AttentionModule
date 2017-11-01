#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random
import os
from visdom import Visdom
import numpy as np

vis = Visdom(port=8097)  #具体参数可参考源码http://best.factj.com/facebookresearch/visdom/tree/master/py/__init__.py
win = None
def plot(name, i, v):
    global win
    if win is None:
        win = vis.line(X=np.array([i]), Y=np.array([v]), opts={'legend':[name]})
    else:
        vis.updateTrace( X=np.array([i]), Y=np.array([v]), win=win, name=name)
