#!/usr/local/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2019/2/20 4:04 PM
# @Author  : L
# @Email   : L862608263@163.com
# @File    : text_outline_extraction.py
# @Software: PyCharm


import os
import cv2

image_folder = os.path.dirname(os.getcwd()) + "/text_outline_extraction/"

for photos in os.listdir(image_folder):

     if str(photos).find(".DS_Store") != -1:
          continue

     if str(photos).find("origin") == -1:
         continue

     image_path = image_folder + photos

     img = cv2.imread(image_path)

     result = img.copy()

     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

     # type cv2.THRESH_BINARY 即二值化，将大于阈值的灰度值设为最大灰度值，小于阈值的值设为0。
     # maxval 当第type类型为CV_THRESH_BINARY和CV_THRESH_BINARY_INV时的最大值
     # thresh  double类型的值，为当前阈值。
     ret, thresh = cv2.threshold(gray, thresh=190, maxval=255, type=cv2.THRESH_BINARY)
     # cv2.imwrite("thresh.jpg", thresh)

     # 腐蚀
     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
     eroded = cv2.erode(thresh, kernel)
     # cv2.imwrite("eroded.jpg", eroded)

     # 膨胀
     # dilation = cv2.dilate(eroded, kernel, iterations=1)
     # cv2.imwrite("dilation.jpg", dilation)

     # 第一个参数是寻找轮廓的图像；
     # 第二个参数表示轮廓的检索模式，有四种（本文介绍的都是新的cv2接口）：
     # cv2.RETR_EXTERNAL表示只检测外轮廓
     # cv2.RETR_LIST检测的轮廓不建立等级关系
     # cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。
     # 如果内孔内还有一个连通物体，这个物体的边界也在顶层。

     # cv2.RETR_TREE建立一个等级树结构的轮廓。

     # 第三个参数method为轮廓的近似办法

     # cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1

     # cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，
     # 例如一个矩形轮廓只需4个点来保存轮廓信息

     contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     color = (0, 255, 0)

     # print(hierarchy)

     for (index, element) in enumerate(contours):

         x, y, w, h = cv2.boundingRect(element)
         next, previous, first_Child, parent = hierarchy[0][index] # 分离树形结构
         if parent == -1 or parent != 0:
             continue
         cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
         temp = result[y:(y + h), x:(x + w)]
         #write_image_path = "result_" + str(index) + ".jpg"
         # cv2.putText(temp, str(index), (0, h), cv2.FONT_HERSHEY_COMPLEX, 1, color)
         #cv2.imwrite(write_image_path, temp)


     cv2.imwrite("result.jpg", img)