#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["SliderExtension, RangeSliderExtension", "CheckButtionsExtension"]

import os
import sys
import cv2
import joblib
import argparse
import numpy as np
from functools import partial
from matplotlib.widgets import (
    Button, Slider, RadioButtons, CheckButtons,
    RectangleSelector, EllipseSelector, LassoSelector)

sys.path.append(".")

from pyutils.figure import Figure
from src.library.range_slider import RangeSlider


class BaseExtension(object):
    klass = None

    def __init__(self, *args, **kwargs):
        self.name = name

    def initialize(self, editor, ax):
        raise NotImplementedError

    def callback(self, editor, event):
        raise NotImplementedError

    def reset(self, editor, event):
        raise NotImplementedError


class SliderExtension(BaseExtension):
    klass = Slider

    def __init__(self, name, valmin, valmax, valinit, **kwargs):
        self.name = name
        self.valmin = valmin
        self.valmax = valmax
        self.value = valinit
        self.kwargs = kwargs

    def initialize(self, editor, ax):
        slider = self.klass(
            ax, self.name, self.valmin, self.valmax,
            valinit=self.value, **self.kwargs)
        slider.on_changed(partial(self.callback, editor))
        return slider

    def callback(self, editor, _event):
        self.value = editor.widgets[self.name].val
        editor.update(use_mask_func=True)

    def reset(self, editor, _event):
        editor.widgets[self.name].reset()


class RangeSliderExtension(BaseExtension):
    klass = RangeSlider

    def __init__(self, name, valmin, valmax, valinit, **kwargs):
        self.name = name
        self.valmin = valmin
        self.valmax = valmax
        self.value = valinit
        self.kwargs = kwargs

    def initialize(self, editor, ax):
        slider = self.klass(
            ax, self.name, self.valmin, self.valmax,
            valinit=self.value, **self.kwargs)
        slider.on_changed(partial(self.callback, editor))
        return slider

    def callback(self, editor, _event):
        self.value = editor.widgets[self.name].val
        editor.update(use_mask_func=True)

    def reset(self, editor, _event):
        editor.widgets[self.name].reset()


class CheckButtionsExtension(BaseExtension):
    klass = CheckButtons

    def __init__(self, name, labels, **kwargs):
        self.name = name
        self.labels = labels
        self.status = [False] * len(self.labels)
        self.kwargs = kwargs

    def initialize(self, editor, ax):
        button = self.klass(
            ax, self.labels, actives=self.status, **self.kwargs)
        button.on_clicked(partial(self.callback, editor))
        for _label in button.labels:
            _label.set_fontsize(10)
        return button

    def callback(self, editor, _event):
        self.status[:] = editor.widgets[self.name].get_status()

    def reset(self, _editor, _event):
        pass
