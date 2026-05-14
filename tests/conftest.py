"""Pytest 全局配置: ifwa 环境必须设置的环境变量。

torch_ifwa device backend autoload 在 ifwa env 内会失败导致 torch import 后
torch.__file__ = None / dir(torch) = []。必须在任何 import torch 之前 disable。

VLLM_VIRTUAL_BACKEND=1: 让 vLLM 在 platform 探测时优先用我们的 VirtualPlatform,
不去 import CUDA runtime (没 libcudart.so 时 import 会挂)。
"""
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("VLLM_VIRTUAL_BACKEND", "1")
