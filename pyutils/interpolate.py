#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["interp1d", "line_normalize", "curve_normalize", "grad"]

import numpy as np
import scipy.interpolate as interp


def interp1d(
        data, x=None, x_start=0.0, x_end=1.0, method=None, **kwargs):
    '''
    interpolate method, see the following link;
    http://scipy.github.io/devdocs/interpolate.html
    '''
    y = np.array(data)
    if x is None:
        x = np.linspace(x_start, x_end, y.shape[0])
    if method is None:
        func = interp.interp1d
        if not("fill_value" in kwargs):
            kwargs["fill_value"] = "extrapolate"
    elif method.lower() == "center":
        func = interp.BarycentricInterpolator
    elif method.lower() == "krogh":
        func = interp.KroghInterpolator
    elif method.lower() == "pchip":
        func = interp.PchipInterpolator
    elif method.lower() == "akima":
        func = interp.Akima1DInterpolator
    elif method.lower() == "cubic":
        func = interp.CubicSpline
    else:
        func = getattr(interp, method)
    return func(x, y, **kwargs)


def line_normalize(func, x=None, y=None, kind=1, eps=1e-10):
    if x is None:
        assert hasattr(func, "x"), "please specify x in argument."
        x = np.array(func.x)
    if y is None:
        assert hasattr(func, "y"), "please specify y in argument."
        y = np.array(func.y)
    y = func(x)
    v = y[1:] - y[:-1]
    ds = np.linalg.norm(v, axis=1) + eps
    ds = ds / sum(ds)
    ds = np.insert(ds, 0, 0)
    fx = interp1d(x, x=np.cumsum(ds), kind=kind, fill_value="extrapolate")
    return lambda _x: func(fx(_x))


def curve_normalize(func, x=None, y=None, kind=1, offset=1.0, eps=1e-10):
    if x is None:
        assert hasattr(func, "x"), "please specify x in argument."
        x = np.array(func.x)
    if y is None:
        y = func(x)
    if hasattr(func, "derivative"):
        df = func.derivative()
        ddf = df.derivative()
    else:
        df = grad(func, dx=1e-2 / len(x))
        ddf = grad(df, dx=1e-2 / len(x))
    v = df(x)
    dv = ddf(x)
    ds = np.abs(np.cross(v, dv))
    ds /= np.linalg.norm(v, axis=1)**3
    ds += (np.max(ds) - np.min(ds)) * offset + eps
    ds = ds[:-1]
    ds = ds / sum(ds)
    ds = np.insert(ds, 0, 0)
    fx = interp1d(x, x=np.cumsum(ds), kind=kind, fill_value="extrapolate")
    return lambda _x: func(fx(_x))


def grad(f, dx=1e-4):
    def function(x, dx=dx):
        _x = np.array(x)
        _dx = dx * 0.5
        return (f(_x + _dx) - f(_x - _dx)) / dx
    return function
