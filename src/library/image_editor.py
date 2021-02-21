#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["ImageEditor"]

import os
import sys
import cv2
import copy
import numpy as np
import matplotlib as mpl
from functools import partial
from matplotlib.widgets import Button, Slider, RadioButtons, CheckButtons
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(".")

from pyutils.figure import Figure
from src.library.base_editor import BaseEditor
from src.library.fmm_estimator import *


class ImageEditor(BaseEditor):
    def __init__(self, image, range_of_interest,
                 mask_func, estimate_p_start, estimate_p_target,
                 resize_rate=1.0, travel_threshold=0.0, wave_coef=2.0, step_width=1.0):
        '''
        Args:
            image_raw (np.ndarray(uint8)): raw image, shape=(H, W, 3)
            mask_raw (np.ndarray(bool)):  mask image, shape=(H, W)
            mask_func (func): mask_func estimator
            estimate_p_start (func): basal point estimator
            estimate_p_target (func): tip point estimator
            travel_threshold (float, optional): Boundary dist, Defaults to 0.
            wave_coef (float, optional): wave speed coefficient. Defaults to 2.
            step_width (float, optional): backward step width. Defaults to 1.
        '''
        self.image_size = tuple(image.shape[:2])
        self.image_edit = np.concatenate([
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            np.full((*image.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
        self.roi_raw = range_of_interest & mask_func(image)
        self.roi_edit = self.roi_raw

        self.mask_func = mask_func
        self.estimate_p_start = estimate_p_start
        self.estimate_p_target = estimate_p_target
        self.extensions = []
        self.extensions += self.estimate_p_start.func.extensions
        self.extensions += self.estimate_p_target.func.extensions
        self.extensions += self.mask_func.extensions

        self._resize_rate = resize_rate
        self._wave_coef = wave_coef
        self._travel_threshold = travel_threshold
        self._step_width = step_width

        self._update_dist_flag = self._update_travel_flag = True

    def estimate(self):
        self.p_start = self.estimate_p_start(
            self.image_edit, self.roi_edit, self.dist_field,
            resize_rate=self._resize_rate)
        self.p_start_init = np.array(self.p_start)
        self.endpoint_num, self.p_target = self.estimate_p_target(
            self.image_edit, self.roi_edit, self.dist_field, self.travel_field,
            resize_rate=self._resize_rate)
        self.p_target_init = np.array(self.p_target)
        self.is_converged = [False] * self.endpoint_num
        self.path = [np.array([]) for _ in range(self.endpoint_num)]
        travel_field = self.travel_field
        if travel_field is None:
            return
        travel_field[self.dist_field < self.travel_threshold] = 1e5
        for _, _p_target in enumerate(self.p_target):
            self.is_converged[_], self.path[_] = estimate_path(
                travel_field, self.p_start, _p_target, self.step_width)

    def retrieve(self):
        return [[_check, _path / self._resize_rate]
                for _check, _path in zip(self.is_converged, self.path)]

    def launch(self):
        if not hasattr(self, "path"):
            self.estimate()
        self.fig = Figure(figsize=(15, 6))
        self.fig.create_grid((1, 3), wspace=0.0, width_ratios=(2, 1, 2))
        self.fig[1].create_grid(
            (4, 3), wspace=0.0, width_ratios=(1, 4.5, 1),
            hspace=0.05, height_ratios=(5, 3.5, 4, 7))

        ax_command = self.fig[1][0, 1]
        ax_command.create_grid((5, 1), hspace=0.0)
        self.button_reset = Button(ax_command[0], 'reset')
        self.button_reset.label.set_fontsize(15)
        self.button_reset.on_clicked(self.reset)
        self.button_start = Button(ax_command[1], 'estimate basal point')
        self.button_start.label.set_fontsize(12)
        self.button_start.on_clicked(self._estimate_start)
        self.button_target = Button(ax_command[2], 'estimate tip point(s)')
        self.button_target.label.set_fontsize(12)
        self.button_target.on_clicked(self._estimate_target)
        self.button_path = Button(ax_command[3], 'estimate skeletal curve')
        self.button_path.label.set_fontsize(10)
        self.button_path.on_clicked(self._estimate_path)
        self.button_end = Button(ax_command[4], 'end')
        self.button_end.label.set_fontsize(15)
        self.button_end.on_clicked(self.terminate)
        self.fig._fig.canvas.mpl_connect('close_event', self.terminate)

        ax_action = self.fig[1][1, 1]
        ax_action.create_grid((2, 1), hspace=0.0, height_ratios=(5, 2))
        self.fig._fig.canvas.mpl_connect(
            'button_press_event', self._on_click_set)
        self.fig._fig.canvas.mpl_connect(
            'motion_notify_event', self._on_click_set)
        self._set_interactive_tool(
            self.fig[0], ax_action[0], ax_action[1])

        ax_fmm_param = self.fig[1][2, 1]
        ax_fmm_param.create_grid(
            (4, 1), hspace=0.0, height_ratios=(2, 1, 1, 1))
        self.button_field = CheckButtons(
            ax_fmm_param[0], ["distance", "travel"])
        for _label in self.button_field.labels:
            _label.set_fontsize(12)
        self.button_field.on_clicked(self.render)
        self.slider_wave = Slider(
            ax_fmm_param[1], r'$\alpha$', 0.0, 5.0,
            valinit=self._wave_coef, valfmt="%.1f")
        self.slider_dist = Slider(
            ax_fmm_param[2], r'$\theta_T$', -5.0, 5.0,
            valinit=self._travel_threshold, valfmt="%.1f")
        self.slider_step = Slider(
            ax_fmm_param[3], r'$\delta$', 0.1, 5.0,
            valinit=self._step_width, valfmt="%.1f")
        self.slider_wave.on_changed(self._on_click_slider)
        self.slider_dist.on_changed(self._on_click_slider)
        self.sliders = [self.slider_wave, self.slider_dist, self.slider_step]

        ax_extension = self.fig[1][3, 1]
        ratios = [1] * len(self.extensions)
        ratios[0] = 2
        self._set_extension(ax_extension, height_ratios=ratios)

        self.fig[0].add_zoom_func()
        rect = plt.Rectangle(
            (0, 0), self.image_size[1], self.image_size[0],
            fc='w', ec='gray', hatch='++', alpha=0.5, zorder=-10)
        self.fig[0].add_patch(rect)
        self.im_image_left = self.fig[0].plot_matrix(
            np.zeros((*self.image_size, 4)), picker=True,
            flip_axis=True, colorbar=False)
        self.fig[0].set_xlim([0, self.image_size[1]])
        self.fig[0].set_ylim([self.image_size[0], 0])
        self.fig[0].set_aspect("equal", "box")
        self.fig[0].set_title(
            "extracted area "
            "(ROI+HSV, left click: write, "
            "left click + shift key: erase)")
        self.im_start, = self.fig[0].plot(
            self.p_start[1], self.p_start[0], "ro", markersize=10)
        self.im_target = [None] * self.endpoint_num
        for _, _p_target in enumerate(self.p_target):
            if _p_target is None:
                self.im_target[_], = self.fig[0].plot(0, 0, "*", markersize=10)
                self.im_target[_].set_visible(False)
                if len(self.path[_]) > 0:
                    self.im_target[_].set_data(*self.path[_][-1][::-1])
                    self.im_target[_].set_visible(True)
            else:
                self.im_target[_], = self.fig[0].plot(
                    _p_target[1], _p_target[0], "*", markersize=10)

        self.fig[2].add_zoom_func()
        self.im_image_right = self.fig[2].plot_matrix(
            self.image_edit[..., :3], picker=True,
            flip_axis=True, colorbar=False)
        self.fig[2].set_xlim([0, self.image_size[1]])
        self.fig[2].set_ylim([self.image_size[0], 0])
        self.fig[2].set_aspect("equal", "box")
        self.fig[2].get_xaxis().set_visible(False)
        self.fig[2].get_yaxis().set_visible(False)
        self.fig[2].set_title("original image")

        self.im_path_left = [None] * self.endpoint_num
        self.im_path_right = [None] * self.endpoint_num
        for _ in range(self.endpoint_num):
            self.im_path_left[_], = self.fig[0].plot(
                [], [], color="pink", marker="+", picker=True)
            self.im_path_right[_], = self.fig[2].plot(
                [], [], color="pink", marker="+", picker=True)
        self.reset()
        self.fig.show()

    def reset(self, event=None):
        self.roi_edit = self.roi_raw & self.mask_func(self.image_edit[..., :3])
        for ext in self.extensions:
            ext.reset(self, event)
        for sld in self.sliders:
            sld.reset()
        self.path = [np.array([]) for _ in range(self.endpoint_num)]
        self.is_converged = [False] * self.endpoint_num
        self.p_start = np.array(self.p_start_init)
        self.p_target = np.array(self.p_target_init)
        self._update_dist_flag = True
        self._update_travel_flag = True
        self.update()

    def update(self, use_mask_func=False):
        if use_mask_func:
            mask = self.mask_func(self.image_edit[..., :3])
            self.roi_edit = self.roi_raw & mask
        self.image_edit[~self.roi_edit, 3] = 0
        self.image_edit[self.roi_edit, 3] = 255
        self.im_image_left.set_data(self.image_edit[::-1])
        self._update_dist_flag = True
        self._update_travel_flag = True
        self.render()

    def render(self, event=None):
        self.im_start.set_data(
            self.p_start[1], self.p_start[0])
        for _id, _p_target in enumerate(self.p_target):
            self.im_target[_id].set_data(_p_target[1], _p_target[0])

        for _i, _path in enumerate(self.path):
            if _path.size > 0:
                self.im_path_left[_i].set_data(_path[:, 1], _path[:, 0])
                self.im_path_right[_i].set_data(_path[:, 1], _path[:, 0])

        check_button = self.button_field.get_status()
        if check_button[0] and (
                self.im_dist is None or self._update_dist_flag):
            dist_field = np.array(self.dist_field)
            if dist_field is not None:
                dist_field[dist_field < 0] = 0
                self.im_dist = self.fig[0].plot_matrix(
                    dist_field, contour=True, flip_axis=False,
                    levels=100, colorbar=False)
        if self.im_dist is not None:
            for _ in self.im_dist.collections:
                _.set_visible(check_button[0])

        if check_button[1] and (
                self.im_travel is None or self._update_travel_flag):
            travel_field = self.travel_field
            if travel_field is not None:
                self.im_travel = self.fig[0].plot_matrix(
                    self.travel_field, contour=True, flip_axis=False,
                    levels=100, colorbar=False)
        if self.im_travel is not None:
            for _ in self.im_travel.collections:
                _.set_visible(check_button[1])
        self.fig._fig.canvas.draw_idle()

    def terminate(self, event=None):
        self.fig.close()

    @property
    def wave_coef(self):
        if hasattr(self, "slider_wave"):
            return self.slider_wave.val
        else:
            return self._wave_coef

    @property
    def travel_threshold(self):
        if hasattr(self, "slider_dist"):
            return self.slider_dist.val
        else:
            return self._travel_threshold

    @property
    def step_width(self):
        if hasattr(self, "slider_step"):
            return self.slider_step.val
        else:
            return self._step_width

    @property
    def dist_field(self):
        if self._update_dist_flag or (not hasattr(self, "_dist_field")):
            self._dist_field = calc_distance(self.roi_edit)
            self._update_dist_flag = False
            if hasattr(self, "im_dist") and (self.im_dist is not None):
                for _ in self.im_dist.collections:
                    _.remove()
            self.im_dist = None
            self._update_travel_flag = True
        if self._dist_field is None:
            return None
        else:
            return np.array(self._dist_field)

    @property
    def travel_field(self):
        if self._update_travel_flag or (not hasattr(self, "_travel_field")):
            self._travel_field = calc_travel(
                self.dist_field, self.p_start,
                self.wave_coef, self.travel_threshold)
            self._update_travel_flag = False
            if hasattr(self, "im_travel") and (self.im_travel is not None):
                for _ in self.im_travel.collections:
                    _.remove()
            self.im_travel = None
        if self._travel_field is None:
            return None
        else:
            return np.array(self._travel_field)

    def _on_click_slider(self, event):
        self._update_travel_flag = True
        self.render()

    def _on_click_set(self, event):
        if not(self.area_bbox.x0 < event.x < self.area_bbox.x1):
            return
        if not(self.area_bbox.y0 < event.y < self.area_bbox.y1):
            return
        if event.key is not None:
            return
        if event.button is None:
            self.drag_id = None
            return

        def _position(ax):
            coords = ax.get_path().vertices
            return ax.get_transform().transform(coords)

        p_mouse = np.array([event.x, event.y])
        p_mouse_data = np.array([event.ydata, event.xdata])
        if event.button == 1 and self.drag_id is None:
            if np.linalg.norm(_position(self.im_start) - p_mouse) < 10.0:
                self.drag_id = -1
            else:
                for _id in range(self.endpoint_num):
                    if np.linalg.norm(
                            _position(self.im_target[_id]) - p_mouse) < 10.0:
                        self.drag_id = _id
                        break
        if self.drag_id == -1:
            self.p_start = p_mouse_data
            self._update_travel_flag = True
        elif self.drag_id is not None:
            self.p_target[self.drag_id] = p_mouse_data
            self.im_target[self.drag_id].set_data(
                self.p_target[self.drag_id][1],
                self.p_target[self.drag_id][0])
        self.render()

    def _estimate_start(self, event=None):
        self.p_start = self.estimate_p_start(
            self.image_edit[..., :3], self.roi_edit,
            self.dist_field, p_start_now=self.p_start,
            resize_rate=self._resize_rate)
        self._update_travel_flag = True
        self.render()

    def _estimate_target(self, event=None):
        self.endpoint_num, self.p_target = self.estimate_p_target(
            self.image_edit[..., :3], self.roi_edit,
            self.dist_field, self.travel_field,
            p_target_now=self.p_target, resize_rate=self._resize_rate)
        self.render()

    def _estimate_path(self, event=None):
        travel_field = self.travel_field
        if travel_field is None:
            for _, _p_target in enumerate(self.p_target):
                self.is_converged[_], self.path[_] = False, np.array([])
        else:
            travel_field[self.dist_field < self.travel_threshold] = 1e5
            for _, _p_target in enumerate(self.p_target):
                self.is_converged[_], self.path[_] = estimate_path(
                    travel_field, self.p_start, _p_target, self.step_width)
        self.render()
