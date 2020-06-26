#!/usr/bin/env python
# coding: utf-8
import os
'''
搭建环境
'''

#安装环境
os.system('python --version')
os.system('pip install --default-timeout=2000 pandas -i https://pypi.tuna.tsinghua.edu.cn/simple')
os.system('pip install --default-timeout=2000 paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple')
os.system('pip install --default-timeout=2000 parl -i https://pypi.tuna.tsinghua.edu.cn/simple')
os.system('pip install --default-timeout=2000 gym  -i https://pypi.tuna.tsinghua.edu.cn/simple')
os.system('pip install --default-timeout=2000 rlschool  -i https://pypi.tuna.tsinghua.edu.cn/simple')
os.system('pip install paddlepaddle-gpu==1.8.2.post107 -i https://mirror.baidu.com/pypi/simple')

# 检查依赖包版本是否正确
os.system('pip list | grep paddlepaddle')
os.system('pip list | grep parl')
os.system('pip list | grep rlschool')

