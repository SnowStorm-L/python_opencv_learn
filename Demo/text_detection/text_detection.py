#!/usr/local/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2019/2/18 2:58 PM
# @Author  : L
# @Email   : L862608263@163.com
# @File    : text_detection.py
# @Software: PyCharm

# import必要的库

import cv2 as opencv

from imutils.object_detection import non_max_suppression

import numpy as np

import time
import os

# 构造参数解析器并解析参数

# ap = argparse.ArgumentParser()
#
# # - image  ：输入图像的路径。
# ap.add_argument("-i", "--image", type=str, default="/Users/l/Desktop/OpenCV/Python_OpenCV_Learn/images/test_image.jpeg",
#                 help="path to input image")
#
# # - east  ：EAST场景文本检测器模型文件路径。
# ap.add_argument("-east", "--east", type=str, help="path to input EAST text detector")
#
# # - min - 置信度  ：确定文本的概率阈值。可选， 默认值= 0.5  。
# ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
#                 help="minimum probability required to inspect a region")
#
# # - width, height  ：调整后的图像宽度,高度 - EAST文本要求输入图像尺寸为32的倍数。可选， 默认值= 320  。
#
# ap.add_argument("-w", "--width", type=int, default=320, help="resized image width (should be multiple of 32)")
#
# ap.add_argument("-e", "--height", type=int, default=320, help="resized image height (should be multiple of 32)")
#
# args = vars(ap.parse_args())

work_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
image_path = work_path + "/images/" + "test_image.jpeg"
# #
# # #
# # args = {"image_patch": image_path}
# #
# # image = opencv.imread(args["image_patch"])
# #
# opencv.namedWindow('image_show', opencv.WINDOW_NORMAL)
# opencv.imshow('image_show', image)
# os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python3.7" to true' ''')
# opencv.waitKey()
# opencv.destroyAllWindows()
# #
# # orig = image.copy()
# #
# # (H, W) = image.shape[:2]
# #
# # print(H, W)

def split_picture(image_path):
    # 以灰度模式读取图片
    image = opencv.imread(image_path, opencv.IMREAD_GRAYSCALE)

    # 将图片的边缘变为白色
    # height, width = image.shape
    # for i in range(width):
    #     image[0, i] = 255
    #     image[height - 1, i] = 255
    # for j in range(height):
    #     image[j, 0] = 255
    #     image[j, width - 1] = 255

    image = preprocess(image)

    opencv.namedWindow('image_show', opencv.WINDOW_NORMAL)
    opencv.imshow('image_show', image)
    os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python3.7" to true' ''')
    opencv.waitKey()
    opencv.destroyAllWindows()


def preprocess(image):
    # 1. Sobel算子，x方向求梯度
    # 此步骤形态学变换的预处理，得到可以查找矩形的图片
    # 参数：输入矩阵、输出矩阵数据类型、设置1、0时差分方向为水平方向的核卷积，设置0、1为垂直方向,ksize：核的尺寸
    sobel = opencv.Sobel(image, opencv.CV_8U, 1, 0, ksize=3)

    # 2. 二值化
    ret, binary = opencv.threshold(sobel, 0, 255, opencv.THRESH_OTSU + opencv.THRESH_BINARY)


    # 7. 存储中间图片
    opencv.imwrite("binary.png", binary)


    return binary

# split_picture(image_path)

import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image file")
args = vars(ap.parse_args())

# load the image from disk
image = opencv.imread(args["image"])

# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
gray = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
gray = opencv.bitwise_not(gray)

# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = opencv.threshold(gray, 0, 255,
                          opencv.THRESH_BINARY | opencv.THRESH_OTSU)[1]

# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = opencv.minAreaRect(coords)[-1]

# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
	angle = -(90 + angle)

# otherwise, just take the inverse of the angle to make
# it positive
else:
	angle = -angle

# rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = opencv.getRotationMatrix2D(center, angle, 1.0)
rotated = opencv.warpAffine(image, M, (w, h),
	flags=opencv.INTER_CUBIC, borderMode=opencv.BORDER_REPLICATE)

# draw the correction angle on the image so we can validate it
opencv.putText(rotated, "Angle: {:.2f} degrees".format(angle),
	(10, 30), opencv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
print("[INFO] angle: {:.3f}".format(angle))
opencv.imshow("Input", image)
opencv.imshow("Rotated", rotated)
opencv.waitKey(0)