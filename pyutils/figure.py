#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["Figure"]

import os
import warnings
import numpy as np
import pandas as pd
from threading import Thread

# importing matplotlib previously for avoiding Qt Errors
import matplotlib

run_on_server = (os.getenv("DISPLAY") is None) and (os.name != 'nt')

if run_on_server:
    matplotlib.use('Agg')

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.axes._subplots import Subplot
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D


warnings.filterwarnings("ignore", category=cbook.mplDeprecation)


# override Subplot class
def __getitem__(self, key, hide_parent=True):
    if hasattr(self, "_grid_spec"):
        if hide_parent:
            self.axis("off")
        spec_key = self._grid_spec[key]
        if not(spec_key in self._spec_list):
            self._ax_dict[spec_key] = self.figure.add_subplot(spec_key)
            self._spec_list.append(spec_key)
        return self._ax_dict[spec_key]
    else:
        return None


def create_grid(self, grid=(1, 1), **kwargs):
    self._ax_dict = {}
    self._spec_list = []
    self._grid_spec = gridspec.GridSpecFromSubplotSpec(
        grid[0], grid[1], subplot_spec=self.get_subplotspec(), **kwargs)


def share_x(self, ax):
    self.get_shared_x_axes().join(self, ax)


def convert_3d(self):
    return self.figure.add_subplot(self.get_subplotspec(), projection="3d")


def line_x(self, x, **kwargs):
    self.axvline(x, 0, 1, **kwargs)


def line_y(self, y, **kwargs):
    self.axhline(y, 0, 1, **kwargs)


def fill_x(self, x0, x1, edgecolor=None, facecolor='pink',
           alpha=0.7, zorder=0, **kwargs):
    y0, y1 = self.get_ylim()
    rect = plt.Rectangle(
        xy=[min(x0, x1), y0], width=abs(x1 - x0), height=y1 - y0,
        edgecolor=edgecolor, facecolor=facecolor, alpha=alpha,
        zorder=zorder, **kwargs)
    self.add_patch(rect)


def fill_y(self, y0, y1, edgecolor=None, facecolor='cyan',
           alpha=0.7, zorder=0, **kwargs):
    x0, x1 = self.get_xlim()
    rect = plt.Rectangle(
        xy=[x0, min(y0, y1)], width=abs(x1 - x0), height=y1 - y0,
        edgecolor=edgecolor, facecolor=facecolor, alpha=alpha,
        zorder=zorder, **kwargs)
    self.add_patch(rect)


def fill_std(self, x, y, std, alpha=0.5, **kwargs):
    self.plot(x, y, **kwargs)
    self.fill_between(x, y - std, y + std, alpha=alpha)


def plot_dataframe(self, df, **kwargs):
    print(df)
    x = df.columns.tolist()
    y = df.index.tolist()
    y.reverse()
    z = df.values
    print(z)
    self.plot_matrix(z, x=x, y=y, **kwargs)


def plot_matrix(self, mat, x=None, y=None, aspect=None, zscale=None,
                vmin=None, vmax=None, ticks_fmt=None, num_label=None,
                colorbar=True, barloc="right", flip_axis=True,
                barsize=0.05, barpad=0.1, contour=False, norm=None,
                extent=None, formatter=None,
                **kwargs):
    if x is None:
        x = np.arange(mat.shape[1])
    if y is None:
        y = np.arange(mat.shape[0])
    x_size, y_size = len(x), len(y)
    if extent is None:
        extent = [0, x_size, 0, y_size]
    # setting aspect of figure
    if aspect == "square":
        aspect = x_size / y_size
    # setting zscale
    if zscale == "log":
        norm = LogNorm(vmin=vmin, vmax=vmax)
        formatter = LogFormatterMathtext()
    elif zscale == "discrete":
        if norm is not None:
            norm = BoundaryNorm(norm, len(norm))
    else:
        norm = None
    # is contour
    if contour:
        im = self.contour(x, y, mat, norm=norm, extent=extent,
                          vmin=vmin, vmax=vmax, **kwargs)
    else:
        im = self.imshow(mat[::-1], norm=norm, extent=extent, aspect=aspect,
                         vmin=vmin, vmax=vmax, **kwargs)
    if flip_axis:
        self.invert_yaxis()
    if colorbar:
        divider = make_axes_locatable(self)
        cax = divider.append_axes(
            barloc, size="{}%".format(100 * barsize), pad=barpad)
        cb = self.figure.colorbar(im, cax=cax, format=formatter)
    if num_label is not None:
        x_skip = int(x_size / num_label)
        y_skip = int(y_size / num_label)
        self.set_xticks(np.arange(x_size)[::x_skip])
        self.set_yticks(np.arange(y_size)[::y_skip])
        if ticks_fmt is None:
            self.set_xticklabels(x[::x_skip])
            self.set_yticklabels(y[::x_skip])
        else:
            xlabels = [ticks_fmt.format(_) for _ in x]
            ylabels = [ticks_fmt.format(_) for _ in y]
            self.set_xticklabels(xlabels[::x_skip])
            self.set_yticklabels(ylabels[::y_skip])
    if colorbar:
        return im, cb
    else:
        return im


def bar_stack(self, data, **kwargs):
    if not hasattr(self, "_ys"):
        self._ys = []
    self._ys.append((data, kwargs))


def bar_plot(self, margin=0.2, space=0):
    ndata = len(self._ys)
    width = (1.0 - (2 * margin + (ndata - 1) * space)) / ndata
    for _i, _y in enumerate(self._ys):
        if type(_y) is tuple:
            _x = [margin + width * (_i + 0.5) + space * _i + _ - 0.5
                  for _ in range(len(_y[0]))]
            self.bar(_x, _y[0], width=width, align='center', **_y[1])
        else:
            x = [margin + width * (_i + 0.5) + space * _i + _ - 0.5
                 for _ in range(len(_y))]
            self.bar(x, _y, width=width, align='center')
    self.set_xticks([_ for _ in range(max([len(y[0]) for y in self._ys]))])
    self._ys = []


def add_zoom_func(self, base_scale=1.5):
    def zoom_reset():
        if not hasattr(zoom_func, "xlim"):
            return
        self.set_xlim(zoom_func.xlim)
        self.set_ylim(zoom_func.ylim)
        self.figure.canvas.draw()

    def zoom_func(event):
        bbox = self.get_window_extent()
        if not(bbox.x0 < event.x < bbox.x1):
            return
        if not(bbox.y0 < event.y < bbox.y1):
            return
        if event.xdata is None or event.ydata is None:
            return
        if not hasattr(zoom_func, "xlim"):
            zoom_func.xlim = self.get_xlim()
            zoom_func.ylim = self.get_ylim()
        if event.button == 2:
            self.zoom_reset()
            return
        cur_xlim = self.get_xlim()
        cur_ylim = self.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        # print(event.button, event.x, event.y, event.xdata, event.ydata)
        if event.button == 'up':
            # deal with zoom in
            scale_factor = base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = 1 / base_scale
        else:
            # deal with something that should never happen
            return
        # set new limits
        self.set_xlim([xdata - (xdata - cur_xlim[0]) / scale_factor,
                       xdata + (cur_xlim[1] - xdata) / scale_factor])
        self.set_ylim([ydata - (ydata - cur_ylim[0]) / scale_factor,
                       ydata + (cur_ylim[1] - ydata) / scale_factor])
        self.figure.canvas.draw()
    self.zoom_reset = zoom_reset
    self.figure.canvas.mpl_connect('scroll_event', zoom_func)
    self.figure.canvas.mpl_connect('button_press_event', zoom_func)


def scientific_ticker(self, axis, sci_format="%1.10e"):
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)

    def g(x, pos):
        return "${}$".format(f._formatSciNotation('%1.10e' % x))

    if "x" in axis:
        self.xaxis.set_major_formatter(mticker.FuncFormatter(g))

    if "y" in axis:
        self.yaxis.set_major_formatter(mticker.FuncFormatter(g))


func_list = [__getitem__, create_grid, share_x, convert_3d,
             line_x, line_y, fill_x, fill_y, fill_std,
             plot_matrix, plot_dataframe, bar_stack, bar_plot, add_zoom_func,
             scientific_ticker]

for _func in func_list:
    setattr(Subplot, _func.__name__, _func)

# configuring Figure class


class Figure(object):
    '''
    Wrapper for matplotlib

    Args:
        num (int, optional): figure id. Defaults to None.
        grid (tuple, optional): grid size (heigth, width). Defaults to (1, 1).
        figsize (tuple, optional):
        figure size (width, height). Defaults to (8, 6).
    '''
    def __init__(self, num=None, grid=(1, 1), **kwargs):
        self._fig = plt.figure(num, **kwargs)
        self._grid_spec = gridspec.GridSpec(grid[0], grid[1])
        self._ax_dict = {}
        self._spec_list = []

    def __getitem__(self, key) -> Subplot:
        spec_key = self._grid_spec[key]
        if not(spec_key in self._spec_list):
            self._ax_dict[spec_key] = self._fig.add_subplot(spec_key)
            self._spec_list.append(spec_key)
        return self._ax_dict[spec_key]

    @staticmethod
    def show(block=True, interval=0.2, tight_layout=True):
        if run_on_server:
            print("no display detected")
        else:
            if tight_layout:
                plt.tight_layout()
            if block:
                plt.show(block=True)
            else:
                plt.show(block=False)
                plt.pause(interval)

    def create_grid(self, grid, **kwargs):
        self._grid_spec = gridspec.GridSpec(grid[0], grid[1], **kwargs)

    def set_figsize(self, *args, **kwargs):
        self._fig.set_size_inches(*args, **kwargs)

    def pause(self, interval, tight_layout=True):
        if tight_layout:
            self._fig.tight_layout()
        plt.pause(interval)

    def clear(self):
        self._fig.clear()

    def close(self):
        plt.close(self._fig)

    def save_pdf(self, file_name, tight_layout=True):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        if tight_layout:
            self._fig.tight_layout()
        pp = PdfPages(file_name)
        self._fig.savefig(pp, format='pdf')
        pp.close()

    def savefig(self, file_name, tight_layout=True, **kwargs):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        if tight_layout:
            self._fig.tight_layout()
        self._fig.savefig(file_name, **kwargs)
