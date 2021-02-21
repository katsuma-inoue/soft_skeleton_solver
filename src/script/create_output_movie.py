#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".")

from pyutils.tqdm import trange, tqdm
from pyutils.figure import Figure
from pyutils.file import basename_without_ext
from pyutils.interpolate import interp1d, line_normalize
from src.library.pre_processor import load_video
from src.library.post_processor import plot_path
from src.library.post_processor import smoothing

parser = argparse.ArgumentParser()
parser.add_argument('video_path', type=str)
parser.add_argument("--cache_name", type=str, default="raw.pkl")
parser.add_argument("--movie_name", type=str, default="movie.mp4")
parser.add_argument("--resize_rate", type=float, default=1.0)
parser.add_argument("--frame_offset", type=int, default=0)
parser.add_argument("--smoothing", action="store_true")
parser.add_argument("--num_padding", type=int, default=5)
parser.add_argument("--resolution", type=int, default=1000)
args = parser.parse_args()

cmap = plt.get_cmap("tab10")

if __name__ == '__main__':
    output_dir = "{}/{}".format(
        os.path.dirname(args.video_path),
        basename_without_ext(args.video_path))
    cache_path = "{}/{}".format(output_dir, args.cache_name)
    movie_path = "{}/{}".format(output_dir, args.movie_name)

    if os.path.exists(cache_path):
        with open(cache_path, mode="rb") as f:
            result_data = joblib.load(f)
    else:
        print("file not found!")

    cap = cv2.VideoCapture(args.video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = np.array([width, height], dtype=int)

    print(tuple(np.array(image_size * args.resize_rate, dtype=int)))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(
        movie_path, fourcc, fps,
        tuple(np.array(image_size * args.resize_rate, dtype=int)))
    for frame_pos in trange(args.frame_offset, length):
        key = cv2.waitKeyEx(30)
        cap.set(1, frame_pos)
        ret, frame = cap.read()
        frame = cv2.resize(
            frame, tuple(np.array(image_size * args.resize_rate, dtype=int)),
            cv2.INTER_LINEAR)
        image = np.array(frame)
        for _i, _res in enumerate(result_data["path"][frame_pos]):
            if _res[0]:
                color = np.array(cmap(_i)) * 255
                color = tuple([int(_) for _ in color[:3][::-1]])
                # color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            if args.smoothing:
                path = smoothing(_res[1], args.num_padding)
                func = line_normalize(
                    interp1d(path, kind=2, axis=0))
                path = func(np.linspace(0, 1, args.resolution))
            else:
                path = _res[1]

            image = plot_path(
                image, path * args.resize_rate, color=color, lw=2)
        # cv2.imshow('frame', image)
        out.write(image)
    out.release()
    cap.release()
    cv2.destroyAllWindows()
