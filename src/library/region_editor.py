#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["RegionEditor"]

import os
import sys
import cv2
import copy
import numpy as np
from functools import partial
from matplotlib.path import Path
from matplotlib.widgets import (
    Button, Slider, RadioButtons, CheckButtons,
    RectangleSelector, EllipseSelector, LassoSelector)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(".")

from pyutils.figure import Figure
from src.library.base_editor import BaseEditor


class RegionEditor(BaseEditor):
    def __init__(self, cap, image_size, resize_rate,
                 range_of_interest, mask_func):
        '''
        Args:
            cap (cv2.VideoCapture): video object
            image_size (tuple(int, int)): [width, heigth]
            resize_rate (float): resize rate
            range_of_interest (np.ndarray(bool)): size=(H, W)
            mask_func (func): function outputing mask
        '''
        self.cap = cap
        self.pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.pos_frame_init = self.pos_frame
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.image_size = tuple(np.array(image_size * resize_rate, dtype=int))
        self.image_size = self.image_size[::-1]
        self.roi_raw = range_of_interest
        self.roi_edit = np.array(range_of_interest)
        self.mask_func = mask_func
        self.extensions = self.mask_func.extensions

    def launch(self):
        self.fig = Figure(figsize=(15, 6))
        self.fig.create_grid((1, 3), wspace=0.0, width_ratios=(2, 1, 2))

        self.fig[1].create_grid(
            (3, 3), wspace=0.0, hspace=0.1, width_ratios=(3, 10, 3))
        ax_command = self.fig[1][0, 1]
        ax_command.create_grid((2, 1), hspace=0.0)
        self.button_reset = Button(ax_command[0], 'reset')
        self.button_reset.label.set_fontsize(15)
        self.button_reset.on_clicked(self.reset)
        self.button_end = Button(ax_command[1], 'end')
        self.button_end.label.set_fontsize(15)
        self.button_end.on_clicked(self.terminate)
        self.fig._fig.canvas.mpl_connect('close_event', self.terminate)

        ax_action = self.fig[1][1, 1]
        ax_action.create_grid((3, 1), hspace=0.0, height_ratios=(5, 2, 2))
        self._set_interactive_tool(
            self.fig[0], ax_action[0], ax_action[1])
        self.slider_frame = Slider(
            ax_action[2], 'frame\npos', 0, self.frame_count - 1,
            valinit=self.pos_frame, valstep=1.0, valfmt='%.0f')
        self.slider_frame.on_changed(self._on_set_image)

        ax_extension = self.fig[1][2, 1]
        self._set_extension(ax_extension)

        # self.fig[0].cla()
        self.fig[0].add_zoom_func()
        rect = plt.Rectangle(
            (0, 0), self.image_size[1], self.image_size[0],
            fc='w', ec='gray', hatch='++', alpha=0.5, zorder=-10)
        self.fig[0].add_patch(rect)
        self.im_image_left = self.fig[0].plot_matrix(
            np.zeros((*self.image_size, 4)), picker=True,
            flip_axis=True, colorbar=False)
        self.fig[0].set_aspect("equal", "box")
        self.fig[0].set_title(
            "ROI of video (left click: write, left click + shift key: erase)")

        # self.fig[2].cla()
        self.fig[2].add_zoom_func()
        rect = plt.Rectangle(
            (0, 0), self.image_size[1], self.image_size[0],
            fc='w', ec='gray', hatch='++', alpha=0.5, zorder=-10)
        self.im_image_right = self.fig[2].plot_matrix(
            np.zeros((*self.image_size, 4)), picker=True,
            flip_axis=True, colorbar=False)
        self.fig[2].set_aspect("equal", "box")
        self.fig[2].add_patch(rect)
        self.fig[2].get_xaxis().set_visible(False)
        self.fig[2].get_yaxis().set_visible(False)
        self.fig[2].set_title("extracted area (ROI+HSV, change HSV ranges)")

        self.reset()
        self.fig.show()

    def reset(self, event=None):
        self.roi_edit = np.array(self.roi_raw)
        for ext in self.extensions:
            ext.reset(self, event)
        self.fig[0].zoom_reset()
        self.fig[2].zoom_reset()
        self.pos_frame = self.pos_frame_init
        self.read()
        self.update()

    def read(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.slider_frame.val))
        ret, frame = self.cap.read()
        if ret is False:
            return
        image = cv2.resize(frame, self.image_size[::-1], cv2.INTER_LINEAR)
        self.image_edit = np.concatenate([
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            np.full((*image.shape[:2], 1), 255, dtype=np.uint8)], axis=2)

    def update(self, **kwargs):
        self.image_edit[~self.roi_edit, 3] = 0
        self.image_edit[self.roi_edit, 3] = 255
        self.im_image_left.set_data(self.image_edit[::-1])
        mask = self.mask_func(self.image_edit[..., :3])
        mask = np.logical_and(mask, self.roi_edit)
        self.image_edit[~mask, 3] = 0
        self.image_edit[mask, 3] = 255
        self.im_image_right.set_data(self.image_edit[::-1])
        self.fig._fig.canvas.draw_idle()

    def terminate(self, event=None):
        self.fig.close()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.pos_frame_init)

    def _on_set_image(self, event=None):
        self.read()
        self.update()
