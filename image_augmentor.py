#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)

import cv2
import numpy as np
import time

from ctypes import *


class ImageStruct(Structure):
    _fields_ =[('width', c_int),  
               ('height', c_int),
               ('channels', c_int),
               ('pixels_buf', POINTER(c_ubyte))]


class ImageAugmentor(object):

    def __init__(self):
        self.lib_aug = cdll.LoadLibrary('./libimageaugmentor.so')

    def __encode_image(self, image):
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]
        pixels_num = width * height * channels
        data_buf = (c_byte * pixels_num)()
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    data_buf[k + j * channels + i * width * channels] = image[i, j, k]
    
        pbuf = cast(data_buf, POINTER(c_ubyte))
        res_st = ImageStruct(width, height, channels, pbuf)
        return res_st
    
    def __decode_image(self, res_st):
        width = res_st.width
        height = res_st.height
        channels = res_st.channels
        image = np.zeros((height, width, channels), dtype=np.uint8)
        for i in range(width * height * channels):
            index_i = i / (width * channels)
            index_j = (i % (width * channels)) / channels
            index_k = i % channels
            image[index_i, index_j, index_k] = res_st.pixels_buf[i]
    
        return image


    def sim_light(self, image, direction):
        image_temp = image.copy()
        st = self.__encode_image(image_temp)
        self.lib_aug.sim_light(st, direction)
        image_temp = self.__decode_image(st)
        return image_temp

    def adjust_color(self, image):
        image_temp = image.copy()
        st = self.__encode_image(image_temp)
        self.lib_aug.adjust_color(st)
        image_temp = self.__decode_image(st)
        return image_temp

    def add_gaussian_noise(self, image):
        image_temp = image.copy()
        st = self.__encode_image(image_temp)
        self.lib_aug.add_gaussian_noise(st)
        image_temp = self.__decode_image(st)
        return image_temp

    def add_pepper_noise(self, image):
        image_temp = image.copy()
        st = self.__encode_image(image_temp)
        self.lib_aug.add_pepper_noise(st)
        image_temp = self.__decode_image(st)
        return image_temp

    def random_blur(self, image):
        image_temp = image.copy()
        st = self.__encode_image(image_temp)
        self.lib_aug.random_blur(st)
        image_temp = self.__decode_image(st)
        return image_temp

    def random_rotate(self, image):
        image_temp = image.copy()
        st = self.__encode_image(image_temp)
        self.lib_aug.random_rotate(st)
        image_temp = self.__decode_image(st)
        return image_temp

    def flip(self, image, method): # -1, 0, 1
        image_temp = image.copy()
        st = self.__encode_image(image_temp)
        self.lib_aug.flip(st, method)
        image_temp = self.__decode_image(st)
        return image_temp


if __name__ == "__main__":
    image = cv2.imread("test.png")
    image_augmentor = ImageAugmentor()

    start_time = time.time()
    for i in range(4):  
        image_temp = image_augmentor.sim_light(image, i)
        # cv2.imshow("main", image_temp)
        # cv2.waitKey()

    for i in range(3):
        image_temp = image_augmentor.flip(image, i - 1)
        # cv2.imshow("main", image_temp)
        # cv2.waitKey()

    for i in range(5):
        image_temp = image_augmentor.adjust_color(image)
        # cv2.imshow("main", image_temp)
        # cv2.waitKey()

    for i in range(5):
        image_temp = image_augmentor.add_gaussian_noise(image)
        # cv2.imshow("main", image_temp)
        # cv2.waitKey()

    for i in range(5):
        image_temp = image_augmentor.add_pepper_noise(image)
        # cv2.imshow("main", image_temp)
        # cv2.waitKey()

    for i in range(5):
        image_temp = image_augmentor.random_blur(image)
        # cv2.imshow("main", image_temp)
        # cv2.waitKey()

    for i in range(5):
        image_temp = image_augmentor.random_rotate(image)
        # cv2.imshow("main", image_temp)
        # cv2.waitKey()

    end_time = time.time()
    print "Use time: %s" % (end_time - start_time)
