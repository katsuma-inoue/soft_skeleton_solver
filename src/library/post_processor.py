#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["smoothing", "plot_path"]

import os
import sys
import cv2
import tqdm
import numpy as np

sys.path.append(".")


def smoothing(data_raw, num_padding):
    data_ext = np.concatenate([
        data_raw[0] - (data_raw[1:num_padding + 1] - data_raw[0])[::-1],
        data_raw,
        data_raw[-1] - (data_raw[-num_padding - 1:-1] - data_raw[-1])[::-1]])
    smoothed = []
    kernel = np.ones(num_padding * 2 + 1) / (num_padding * 2 + 1)
    for _ in range(data_ext.shape[1]):
        smoothed.append(
            np.convolve(data_ext[:, _], kernel, mode='valid'))
    return np.array(smoothed).T


def plot_path(image, path_list, color=(0, 0, 255), lw=2):
    res = np.array(image)
    path_list = list(map(tuple, np.array(path_list).astype(int)))
    for _pre, _post in zip(path_list[0:-1], path_list[1:]):
        res = cv2.line(res, _pre[::-1], _post[::-1], color, lw)
    return res
