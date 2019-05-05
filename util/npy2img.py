from __future__ import division

import os
from os.path import join

import cv2 as cv
import numpy as np


def load_npy(root, folder='npy_5', drive_num=1, idx=1):
    '''load .npy file
    Args:
        root: root path
        folder: folder name
        drive_num: index of drive part
        idx: index of npy
    Returns:
        file_name: full name of npy
        data: numpy array data
    '''
    full_name = '2011_09_26_%04d_%010d.npy' % (drive_num, idx)
    full_path = join(root, folder, full_name)
    return full_name, np.load(full_path)


def summary(root, folder):
    '''calculate the accumulative pixels of each class
    Args:
        root: root path
        folder:  contain .npy file
    '''
    path = join(root, folder)
    pixel_sum = {'0': 0, '1': 0, '2': 0, '3': 0}
    image_sum = {'0': 0, '1': 0, '2': 0, '3': 0}
    for file in os.listdir(path):
        npy = np.load(join(path, file))
        label = npy[:,:, 5]
        rgb_mask = (npy[:, :, 6] > 0) | (
            npy[:,:, 7] > 0) | (npy[:,:, 8] > 0)
        label = label * rgb_mask
        for i in range(4):
            pixel_sum[str(i)] += np.sum(label == i)
            if np.sum(label == i):
                image_sum[str(i)] += 1
        print('process:',file)
    print(pixel_sum)
    print(image_sum)
    

def save_label(npy_data, save=True, plot=False, resize = (1242,375)):
    '''transfer label from npy data into colored jpg
    Args:
        npy_data: numpy format data with the shape of 64x512x10
            (x,y,z,intensity,depth,label,r,g,b,mask)
        plot: whether to plot the picture
        resize: tuple, w x h
    '''
    # ('Car': 0): red  ('Pedestrian': 0): green  ('Cyclist': 0): blue
    COLOR_MAP = np.array([[0.00,  0.00,  0.00],
                          [0.0,  0.0,  0.99],
                          [0.0,  0.99,  0.0],
                          [0.99,  0.0,  0.0]])
    rgb_mask = (npy_data[:, :, 6] > 0) | (npy_data[:, :, 7] > 0) | (npy_data[:, :, 8] > 0)
    label = npy_data[:, :, 5] * rgb_mask
    out = np.zeros((label.shape[0], label.shape[1], 3))
    for i in range(4):
        out[label == i] = COLOR_MAP[i]
    if resize:
        out = cv.resize(out, resize)
    if plot:
        cv.imshow('color label', out)
    if save:
        cv.imwrite('color_label.jpg', out*255)


def save_layer(npy_data, save=True, plot=False, resize=(1242,375)):
    '''transfer each layer in npy data into jpg
    Args:
        npy_data: numpy format data with shape of 64x512x10
        resize: tuple, w x h
    '''
    rgb_mask = (npy_data[:, :, 6] > 0) | (npy_data[:,:, 7] > 0) | (npy_data[:,:, 8] > 0)
    for i in range(npy_data.shape[2]):
        layer = npy_data[:, :, i] * rgb_mask
        out = cv.resize(layer, (1242, 375))
        out = (out-out.min())/(out.max()-out.min())*255
        if plot:
            cv.imshow('layer%d' % i, out)
        if save:
            cv.imwrite('layer_%d.jpg'%i, out)

if __name__ == "__main__":
    root_path = r'C:\Users\evql\Desktop\VscodeProjects\sphereProjection\data\source_5_m5_3'

    _, npy_data = load_npy(root_path, folder='npy_xyzidlrgbm', drive_num=5, idx=142)

    save_label(npy_data)
    save_layer(npy_data)

    pass
