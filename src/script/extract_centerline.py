#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import joblib
import argparse
import numpy as np
from functools import partial

sys.path.append(".")

from src.library.extention import (
    SliderExtension, RangeSliderExtension, CheckButtionsExtension)
from src.library.main_process import main_process
from src.library.post_processor import smoothing
from src.library.pre_processor import load_video

from pyutils.figure import Figure
from pyutils.file import basename_without_ext
from pyutils.interpolate import interp1d, line_normalize

parser = argparse.ArgumentParser()
parser.add_argument("video_path", type=str)
parser.add_argument("--cache_name", type=str, default="raw.pkl")
parser.add_argument("--reset_cache", action="store_true")
# judge threshold
parser.add_argument("--warp_threshold", type=float, default=np.inf)
# main_process arguments
parser.add_argument("--skeleton_num", type=int, default=1)
parser.add_argument("--frame_offset", type=int, default=0)
parser.add_argument("--frame_end", type=int, default=None)
parser.add_argument("--travel_threshold", type=float, default=-1.0)
parser.add_argument("--wave_coef", type=float, default=0.5)
parser.add_argument("--step_width", type=float, default=1.0)
parser.add_argument("--resize_rate", type=float, default=1.0)
parser.add_argument("--show_incomplete_frame", action="store_true")
parser.add_argument("--skip_editor", action="store_true")
args = parser.parse_args()


def create_mask(frame):
    hsv_min = tuple(int(_.value[0]) for _ in create_mask.extensions)
    hsv_max = tuple(int(_.value[1]) for _ in create_mask.extensions)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, hsv_min, hsv_max)
    return mask > 0


hue = RangeSliderExtension(
    "H", 0, 255, (0, 255), valfmt='(%.0f,%.0f)', valstep=1.0)
sat = RangeSliderExtension(
    "S", 0, 255, (0, 255), valfmt='(%.0f,%.0f)', valstep=1.0)
val = RangeSliderExtension(
    "V", 0, 255, (0, 255), valfmt='(%.0f,%.0f)', valstep=1.0)
create_mask.extensions = [hue, sat, val]


def estimate_p_start(
        frame, mask, dist_field,
        frame_pos=0, pre_result=None, p_start_now=None,
        resize_rate=1.0, **kwargs):
    hidden_area = ~mask
    fix_h, fix_w = estimate_p_start.extensions[0].status
    radius = estimate_p_start.extensions[1].value
    p_start_pre = None
    if p_start_now is None:
        for _is_done, _path in pre_result:
            if _is_done and len(_path) > 0:
                p_start_pre = np.array(_path[0]) * resize_rate
                break
    else:
        p_start_pre = p_start_now
    if p_start_pre is not None:
        y, x = p_start_pre.astype(int)
        height, width = dist_field.shape
        ys, xs = np.ogrid[-y:height - y, -x:width - x]
        dist_circle = (xs * xs + ys * ys) > radius ** 2
        hidden_area = np.logical_or(hidden_area, dist_circle)
        if fix_h:
            offset_field = np.full(dist_field.shape, True)
            offset_field[int(p_start_pre[0]), :] = False
            hidden_area = np.logical_or(hidden_area, offset_field)
        if fix_w:
            offset_field = np.full(dist_field.shape, True)
            offset_field[:, int(p_start_pre[1])] = False
            hidden_area = np.logical_or(hidden_area, offset_field)
    dist_mask = np.ma.MaskedArray(dist_field, hidden_area)
    p_start = np.array(np.unravel_index(
        dist_mask.argmax(), dist_mask.shape), dtype=float)
    return p_start


fix_b = CheckButtionsExtension("fix_p_start", [
    "fix h of basal point", "fix w of basal point"])
rad_b = SliderExtension(r"$\epsilon_b$", 0, 1000, 1000, valfmt='%.0f')
estimate_p_start.extensions = [fix_b, rad_b]


def estimate_p_target(
        frame, mask, dist_field, travel_field,
        frame_pos=0, pre_result=None, p_target_now=None,
        resize_rate=1.0, **kwargs):
    radius = estimate_p_target.extensions[0].value
    margin = estimate_p_target.extensions[1].value
    p_target = []
    for _, (_is_done, _path) in enumerate(pre_result):
        new_mask = dist_field < margin
        pos = None
        if p_target_now is not None:
            pos = p_target_now[_]
        elif _is_done:
            pos = _path[-1] * resize_rate
        if (pos is not None) and (travel_field is not None):
            y_pre, x_pre = pos
            y_pre, x_pre = int(y_pre), int(x_pre)
            height, width = dist_field.shape
            ys, xs = np.ogrid[-y_pre:height - y_pre, -x_pre:width - x_pre]
            dist_mask = (xs * xs + ys * ys) > radius ** 2
            new_mask = np.logical_or(dist_mask, new_mask)
            if not np.all(new_mask):
                travel_masked = np.ma.MaskedArray(travel_field, new_mask)
                pos_new = np.array(np.unravel_index(
                    travel_masked.argmax(),
                    travel_masked.shape), dtype=float)
                if pos_new[0] == 0 and pos_new[1] == 0:
                    p_target.append(pos)
                else:
                    p_target.append(pos_new)
            else:
                p_target.append(pos)
        else:
            p_target.append(np.array(dist_field.shape) * 0.25)
    p_target = np.array(p_target)
    return p_target.shape[0], p_target


rad_t = SliderExtension(r"$\epsilon_t$", 0, 1000, 50, valfmt='%.0f')
mar_t = SliderExtension(r"$\theta_\min$", -5.0, 5.0, 0.0, valfmt='%.1f')
estimate_p_target.extensions = [rad_t, mar_t]


def judge_path(frame_pos, cur_result, pre_result):
    def normalize(_path):
        path_now = smoothing(_path, 5)
        func = line_normalize(
            interp1d(path_now, kind=1, axis=0))
        path_now = func(np.linspace(0, 1, 200))
        return path_now

    judge_result = []
    for _res, _pre in zip(cur_result, pre_result):
        flag, error_message = True, ""
        if not _res[0]:
            if _res[1].size == 0:
                error_message += "untracked frame!"
                flag = False
            elif frame_pos == args.frame_offset:
                error_message += "initial frame!"
                flag = False
            else:
                flag = False
        elif frame_pos > 0 and _pre[0]:
            diff = normalize(_res[1]) - normalize(_pre[1])
            # norm = np.mean(np.linalg.norm(diff, axis=1))
            norm = np.max(np.linalg.norm(diff, axis=1))
            print("norm_max = {:.6f}".format(norm))
            if norm > args.warp_threshold:
                error_message += "warp detected!"
                flag = False
        if not flag:
            print("\x1b[2Kframe {}, {}".format(frame_pos, error_message))
        judge_result.append(flag)
    return judge_result


if __name__ == '__main__':
    output_dir = "{}/{}".format(
        os.path.dirname(args.video_path),
        basename_without_ext(args.video_path))
    output_path = "{}/{}".format(output_dir, args.cache_name)
    os.makedirs(output_dir, exist_ok=True)
    if not args.reset_cache and os.path.exists(output_path):
        with open(output_path, mode="rb") as f:
            result_data = joblib.load(f)
        if args.show_incomplete_frame:
            frame_unfinished = []
            for _frame in range(len(result_data["path"])):
                if not np.all([_[0] for _ in result_data["path"][_frame]]):
                    if _frame >= args.frame_offset:
                        frame_unfinished.append(_frame)
            print(frame_unfinished)
            sys.exit()
    else:
        cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.video_path))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        result_data = {}
        result_data["path"] = [
            [[False, np.array([])] for _ in range(args.skeleton_num)]
            for _ in range(length)]

    result_data = main_process(
        args.video_path, result_data, create_mask, estimate_p_start,
        estimate_p_target, judge_path, resize_rate=args.resize_rate,
        frame_offset=args.frame_offset, frame_end=args.frame_end,
        wave_coef=args.wave_coef, step_width=args.step_width,
        travel_threshold=args.travel_threshold,
        skip_editor=args.skip_editor)

    with open(output_path, mode="wb") as f:
        joblib.dump(result_data, f, compress=True)

    cnt_is_done = sum([sum(map(lambda _: _[0], _res)) for _res in result_data["path"]])
    cnt_total = len(result_data["path"]) * len(result_data["path"][0])
    print("\x1b[2K# of complete frames {} / {}".format(cnt_is_done, cnt_total))
