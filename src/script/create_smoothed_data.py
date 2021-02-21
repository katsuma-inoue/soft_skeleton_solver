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

import src.library.style
from src.library.analysis import *
from src.library.pre_processor import load_video
from src.library.post_processor import plot_path
from src.library.post_processor import smoothing

parser = argparse.ArgumentParser()
parser.add_argument('video_path', type=str)
parser.add_argument("--cache_name", type=str, default="raw.pkl")
parser.add_argument("--save_name", type=str, default="smoothed.pkl")
# main_process arguments
parser.add_argument("--num_padding", type=int, default=5)
parser.add_argument("--resolution", type=int, default=1000)
args = parser.parse_args()


cmap = plt.get_cmap("tab10")


if __name__ == '__main__':
    output_dir = "{}/{}".format(
        os.path.dirname(args.video_path),
        basename_without_ext(args.video_path))
    cache_path = "{}/{}".format(output_dir, args.cache_name)
    save_path = "{}/{}".format(output_dir, args.save_name)
    fig_path = "{}/{}.pdf".format(
        output_dir, basename_without_ext(args.save_name))

    if os.path.exists(cache_path):
        with open(cache_path, mode="rb") as f:
            result_data = joblib.load(f)
    else:
        print("file not found!")

    frame_num = len(result_data["path"])
    endpoint_num = len(result_data["path"][0])
    print(endpoint_num)
    result_smoothed = {_: np.full(
        (frame_num, args.resolution, 2), np.nan) for _ in range(endpoint_num)}
    print("\nsmoothing begin!")
    for _frame_pos in trange(frame_num):
        for _id, (_is_valid, _path) in enumerate(result_data["path"][_frame_pos]):
            if _path.shape[0] >= args.num_padding:
                path_now = smoothing(_path, args.num_padding)
                func = line_normalize(
                    interp1d(path_now, kind=2, axis=0))
                path_now = func(np.linspace(0, 1, args.resolution))
                result_smoothed[_id][_frame_pos] = path_now

    with open(save_path, mode="wb") as f:
        joblib.dump(result_smoothed, f, compress=True)

    fig = Figure(figsize=(10, 6))
    fig.create_grid((len(result_smoothed), 1))
    for _id, _path in result_smoothed.items():
        print(_path.shape)
        fig[_id].create_grid((2, 1), hspace=0.0)
        fig[_id][0].plot_matrix(
            _path[:, :, 0].T, aspect="auto", cmap="viridis")
        fig[_id][0].set_xticklabels([])
        fig[_id][0].set_yticklabels([])
        fig[_id][1].plot_matrix(
            _path[:, :, 1].T, aspect="auto", cmap="magma")
        fig[_id][1].set_yticklabels([])

    fig.savefig(fig_path)
    Figure.show()
