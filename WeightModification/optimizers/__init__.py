# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/10/9 20:24

"""
from https://github.com/Yonghongwei/Gradient-Centralization
"""
__all__ = ['Adam', 'AdamW', 'SGD', 'RAdam', 'PlainRAdam', 'Ranger']

from .Adam import Adam, AdamW
from .SGD import SGD
from .RAdam import RAdam, PlainRAdam
from .Ranger import Ranger