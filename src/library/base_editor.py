#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["BaseEditor"]

import os
import sys
import cv2
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
from src.library.range_slider import RangeSlider


class BaseEditor(object):
    pen_size = 20.0

    def __init__(self, *args, **kwargs):
        pass

    def launch(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def terminate(self, *args, **kwargs):
        raise NotImplementedError

    def _set_interactive_tool(self, ax_area, ax_button, ax_slider):
        self.drag_id = None
        self.area_bbox = ax_area.get_window_extent()

        self.selector_pen = patches.Circle(
            xy=(0, 0), edgecolor='black',
            radius=0.0, fc='gray',
            alpha=0.5, zorder=5)
        ax_area.add_patch(self.selector_pen)
        self.selector_pen.set_visible(True)

        self.fig._fig.canvas.mpl_connect(
            'button_press_event', self._on_click_pen)
        self.fig._fig.canvas.mpl_connect(
            'motion_notify_event', self._on_click_pen)

        self.selector_rect = RectangleSelector(
            ax_area, self._on_select_rect,
            drawtype='box', button=1,
            rectprops=dict(
                facecolor='gray', edgecolor='black',
                alpha=0.5, fill=True, zorder=5),
            state_modifier_keys=dict(
                move=' ', clear='escape',
                square='control', center='space'))
        self.selector_rect.visible = False

        self.selector_ellipse = EllipseSelector(
            ax_area, self._on_select_ellipse,
            drawtype='box', button=1,
            rectprops=dict(
                facecolor='gray', edgecolor='black',
                alpha=0.5, fill=True, zorder=5),
            state_modifier_keys=dict(
                move=' ', clear='escape',
                square='control', center='space'))
        self.selector_ellipse.visible = False

        def _press(ls, event):
            ls.verts = [ls._get_data(event)]
            ls.line.set_visible(ls.visible)

        def _release(ls, event):
            if ls.verts is not None:
                ls.verts.append(ls._get_data(event))
                ls.onselect(ls.verts, event)
            ls.line.set_data([[], []])
            ls.line.set_visible(False)
            ls.verts = None

        LassoSelector._press = _press
        LassoSelector._release = _release
        self.selector_lasso = LassoSelector(
            ax_area, onselect=self._on_select_lasso,
            button=1, useblit=False,
            lineprops=dict(linewidth=2.0, color="gray", zorder=1))
        self.selector_lasso.visible = False

        self.button_radio = RadioButtons(
            ax_button, ["pen", "rectangle", "ellipse", "lasso"])
        for _label in self.button_radio.labels:
            _label.set_fontsize(12)
        self.button_radio.on_clicked(self._on_change_radio)

        self.slider_radius = Slider(
            ax_slider, 'pen\nsize', 0.0, 50.0,
            valinit=BaseEditor.pen_size, valfmt='%.1f')
        self.slider_radius.on_changed(self._on_change_radius)

    def _set_extension(self, ax_extension, **kwargs):
        ax_extension.create_grid(
            (len(self.extensions), 1), hspace=0.0, **kwargs)
        self.widgets = {}
        for _, ext in enumerate(self.extensions):
            self.widgets[ext.name] = ext.initialize(self, ax_extension[_])

    def _on_change_radius(self, event):
        BaseEditor.pen_size = self.slider_radius.val

    def _on_change_radio(self, label):
        if label == "pen":
            self.selector_pen.set_visible(True)
        else:
            self.selector_pen.set_visible(False)
        if label == "rectangle":
            self.selector_rect.visible = True
        else:
            self.selector_rect.visible = False
        if label == "ellipse":
            self.selector_ellipse.visible = True
        else:
            self.selector_ellipse.visible = False
        if label == "lasso":
            self.selector_lasso.visible = True
        else:
            self.selector_lasso.visible = False
        self.fig._fig.canvas.draw_idle()

    def _on_click_pen(self, event=None):
        if self.button_radio.value_selected != "pen":
            return
        if self.drag_id is not None:
            self.selector_pen.set_visible(False)
            return
        if not(self.area_bbox.x0 < event.x < self.area_bbox.x1):
            return
        if not(self.area_bbox.y0 < event.y < self.area_bbox.y1):
            return
        self.selector_pen.set_visible(True)
        self.selector_pen.center = (event.xdata, event.ydata)
        self.selector_pen.radius = self.slider_radius.val
        self.fig._fig.canvas.draw_idle()

        if event.button != 1:
            return
        pos_x, pos_y = int(event.xdata), int(event.ydata)
        _r = self.slider_radius.val
        _y, _x = np.ogrid[
            -pos_y:(self.image_size[0] - pos_y),
            -pos_x:(self.image_size[1] - pos_x)]
        _mask = _x * _x + _y * _y <= _r * _r
        if event.key == "shift":
            self.roi_edit[_mask] = False
        else:
            self.roi_edit[_mask] = True
        self.update()

    def _on_select_rect(self, eclick, erelease):
        if self.button_radio.value_selected != "rectangle":
            return
        if self.drag_id is not None:
            return
        xslice = [min(max(_, 0), self.image_size[1])
                  for _ in [eclick.xdata, erelease.xdata]]
        xslice = tuple(sorted(map(int, xslice)))
        yslice = [min(max(_, 0), self.image_size[0])
                  for _ in [eclick.ydata, erelease.ydata]]
        yslice = tuple(sorted(map(int, yslice)))
        if erelease.key == "shift":
            self.roi_edit[slice(*yslice), slice(*xslice)] = False
        else:
            self.roi_edit[slice(*yslice), slice(*xslice)] = True
        self.update()

    def _on_select_ellipse(self, eclick, erelease):
        if self.button_radio.value_selected != "ellipse":
            return
        if self.drag_id is not None:
            return
        xslice = [eclick.xdata, erelease.xdata]
        xslice = tuple(sorted(map(int, xslice)))
        yslice = [eclick.ydata, erelease.ydata]
        yslice = tuple(sorted(map(int, yslice)))
        pos_x = int((xslice[0] + xslice[1]) * 0.5)
        pos_y = int((yslice[0] + yslice[1]) * 0.5)
        rad_x = int((xslice[1] - xslice[0]) * 0.5)
        rad_y = int((yslice[1] - yslice[0]) * 0.5)
        if rad_x == 0.0 or rad_y == 0.0:
            return
        _y, _x = np.ogrid[
            -pos_y:(self.image_size[0] - pos_y),
            -pos_x:(self.image_size[1] - pos_x)]
        _mask = (_x / rad_x) ** 2 + (_y / rad_y) ** 2 <= 1.0
        if erelease.key == "shift":
            self.roi_edit[_mask] = False
        else:
            self.roi_edit[_mask] = True
        self.update()

    def _on_select_lasso(self, verts, event=None):
        if self.button_radio.value_selected != "lasso":
            return
        if self.drag_id is not None:
            return
        path = Path(verts)
        if not hasattr(self, "coords"):
            self.coords = np.array(np.meshgrid(
                *list(map(np.arange, self.image_size)), indexing="ij"))
            self.coords = self.coords.transpose((1, 2, 0)).reshape(-1, 2)
        _mask = path.contains_points(
            self.coords[:, ::-1]).reshape(*self.image_size)
        if event.key == "shift":
            self.roi_edit[_mask] = False
        else:
            self.roi_edit[_mask] = True
        self.update()
