# -*- coding: utf-8 -*-

import cv2

image = cv2.imread('result/001/000101.jpg')
resized_image = cv2.resize(image, (224, 224))
