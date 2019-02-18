#!/usr/local/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2019/2/18 5:52 PM
# @Author  : L
# @Email   : L862608263@163.com
# @File    : correct_skew.py
# @Software: PyCharm

import os
import cv2

image_folder = os.getcwd() + "/images/"

print('image_path: ' + image_folder)

for photos in os.listdir(image_folder):
    print(image_folder + photos)