# ==============================================================================
# Binaries and/or source for the following packages or projects are presented under one or more of the following open
# source licenses:
# utils.py       The OpenLane Dataset Authors        Apache License, Version 2.0
#
# Contact simachonghao@pjlab.org.cn if you have any issue
# 
# See:
# https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection/blob/master/tools/utils.py
#
# Copyright (c) 2022 The OpenLane Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-
import argparse
import errno
import os
import numpy as np
from scipy.interpolate import interp1d


def define_args():
    parser = argparse.ArgumentParser(description='3D lane evaluation')
    # Paths settings
    parser.add_argument('--dataset_dir', type=str, help='The path saving actual data')
    parser.add_argument('--pred_dir', type=str, help='The path of prediction result')
    parser.add_argument('--test_list', type=str, help='The path of test list txt')
    # parser.add_argument('--images_dir', type=str, help='The path saving dataset images')

    return parser
