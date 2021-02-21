#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["main_process"]

import os
import sys
import cv2
import copy
import numpy as np
from functools import partial

sys.path.append(".")

from pyutils.figure import Figure
from src.library.image_editor import ImageEditor
from src.library.region_editor import RegionEditor
from src.library.post_processor import plot_path


def main_process(video_path, result_data,
                 create_mask, estimate_p_start, estimate_p_target, judge_path,
                 resize_rate=1.0, frame_offset=0, frame_end=None,
                 travel_threshold=0.0, wave_coef=0.5, step_width=1.0,
                 skip_editor=False):
    frame_pos = frame_offset
    pause_flag = False
    run_judge_flag = True
    run_estimate_flag = True
    launch_region_editor = False
    launch_image_editor = True

    cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(video_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = np.array([width, height], dtype=int)
    resize_size = tuple(np.array(image_size * resize_rate, dtype=int))
    if frame_end is None:
        frame_end = length - 1

    def on_trackbar(val):
        global frame_pos
        frame_pos = int(val)
        if frame_pos < frame_offset or frame_pos > frame_end:
            frame_pos = frame_offset

    # initializing display field
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    _ret, frame = cap.read()
    frame = cv2.resize(frame, resize_size, cv2.INTER_LINEAR)
    cv2.imshow('frame', frame)
    cv2.createTrackbar('time', 'frame', frame_offset, frame_end, on_trackbar)

    if not("roi" in result_data) or (result_data["roi"].shape != frame.shape[:2]):
        result_data["roi"] = np.full(frame.shape[:2], True)
        launch_region_editor = True

    if np.all([_[0] for _ in result_data["path"][frame_pos]]):
        launch_image_editor = False

    try:
        while True:
            key = cv2.waitKeyEx(30)
            if key == 27 or key == ord("q"):
                break
            if key == 32:
                pause_flag = not pause_flag

            if launch_region_editor:
                editor = RegionEditor(
                    cap, image_size, resize_rate, result_data["roi"], create_mask)
                editor.launch()
                result_data["roi"][:] = editor.roi_edit
                launch_region_editor = False

            if run_judge_flag:
                judge = judge_path(
                    frame_pos, result_data["path"][frame_pos],
                    result_data["path"][frame_pos - 1])
                for _i, _judge in enumerate(judge):
                    result_data["path"][frame_pos][_i][0] = _judge
                run_judge_flag = False
            is_done = np.all([_[0] for _ in result_data["path"][frame_pos]])

            def create_editor():
                func_start = partial(
                    estimate_p_start,
                    pre_result=result_data["path"][frame_pos - 1],
                    frame_pos=frame_pos)
                func_target = partial(
                    estimate_p_target,
                    pre_result=result_data["path"][frame_pos - 1],
                    frame_pos=frame_pos)
                return ImageEditor(
                    frame, result_data["roi"], create_mask, func_start, func_target,
                    resize_rate, travel_threshold, wave_coef, step_width)

            editor, result, judge = None, None, None
            if (not is_done) and run_estimate_flag:
                editor = create_editor()
                editor.estimate()
                result = editor.retrieve()
                judge = judge_path(
                    frame_pos, result, result_data["path"][frame_pos - 1])
                launch_image_editor = (not skip_editor) and (not np.all(judge))
                run_estimate_flag = False

            if launch_image_editor:
                if editor is None:
                    editor = create_editor()
                editor.launch()
                result = editor.retrieve()
                travel_threshold = editor.travel_threshold
                wave_coef = editor.wave_coef
                step_width = editor.step_width
                judge = judge_path(
                    frame_pos, result, result_data["path"][frame_pos - 1])
                launch_image_editor = False
                run_estimate_flag = False
                pause_flag = True

            if not(result is None or judge is None):
                for _i, (_judge, _res) in enumerate(zip(judge, result)):
                    result_data["path"][frame_pos][_i][0] = _judge
                    result_data["path"][frame_pos][_i][1] = _res[1]
                is_done = np.all([_[0] for _ in result_data["path"][frame_pos]])

            # plot_result
            for _valid, _path in result_data["path"][frame_pos]:
                if _valid:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                image = plot_path(
                    frame, _path * resize_rate, color=color, lw=2)
            cv2.imshow('frame', image)

            frame_pos_pre = frame_pos
            if pause_flag:
                if key == ord('x'):
                    frame_pos += 1
                elif key == ord('z'):
                    frame_pos -= 1
                elif key == ord('r'):
                    frame_pos = frame_offset
                elif key == ord('v'):
                    fig = Figure()
                    fig[0].plot_matrix(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), picker=True,
                        flip_axis=True, colorbar=False)
                    fig[0].add_zoom_func()
                    for _valid, _path in result_data["path"][frame_pos]:
                        fig[0].plot(
                            _path[:, 1], _path[:, 0],
                            color="green", marker="+", picker=True)
                    Figure.show()
                    fig.close()
                elif key == 13:  # enter key
                    launch_image_editor = True
                elif key == ord('m'):
                    launch_region_editor = True
            else:
                frame_pos += 1

            if frame_pos_pre != frame_pos:
                launch_image_editor = False
                launch_region_editor = False
                run_judge_flag = True
                run_estimate_flag = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                frame = cv2.resize(frame, resize_size, cv2.INTER_LINEAR)
                cv2.setTrackbarPos('time', 'frame', frame_pos)
            if frame_pos < frame_offset or frame_pos > frame_end:
                frame_pos = frame_offset
    except KeyboardInterrupt:
        pass
    print("stop program")
    cap.release()
    cv2.destroyAllWindows()
    return result_data
