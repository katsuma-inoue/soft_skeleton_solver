#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["load_video", "longest_continuous_array"]

import os
import sys
import cv2
import tqdm
import numpy as np


def load_video(video_path):
    print("loading {} begin!".format(video_path))
    cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(video_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("video size {}".format((length, height, width)))
    raw_data = np.zeros((length, height, width, 3), dtype=np.uint8)
    iterator = tqdm.trange(length)
    for _id in iterator:
        try:
            ret, frame = cap.read()
            raw_data[_id] = frame
        except TypeError:
            iterator.close()
            length = _id
            print("invalid value detected! [{}] {}".format(_id, frame))
            break
    raw_data.resize(length, height, width, 3)
    print("loading {} complete!".format(video_path))
    return raw_data


def longest_continuous_array(arr):
    len_arr = len(arr)
    arr = np.array(arr)
    bin_count = np.bincount((~arr).cumsum()[arr])
    max_id = bin_count.argmax()
    bin_id = np.cumsum(bin_count)
    base_arr = np.arange(len_arr)[arr]
    if max_id == 0:
        return base_arr[slice(0, bin_id[max_id])]
    else:
        return base_arr[slice(bin_id[max_id - 1], bin_id[max_id])]
