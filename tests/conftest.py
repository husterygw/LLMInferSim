"""Pytest 全局配置: ifwa 环境必须设置的环境变量。

torch_ifwa device backend autoload 在 ifwa env 内会失败导致 torch import 后
torch.__file__ = None / dir(torch) = []。必须在任何 import torch 之前 disable。
"""
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
