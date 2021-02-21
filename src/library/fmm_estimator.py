#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "vector_field", "nabla", "calc_distance", "calc_travel", "estimate_path"]

import os
import sys
import time
import skfmm
import numpy as np
import scipy.interpolate as interp


def vector_field(mat):
    x = np.array(range(mat.shape[0]))
    y = np.array(range(mat.shape[1]))
    f = interp.interp2d(y, x, mat, kind='linear')
    return lambda p: f(p[1], p[0])[0]


def nabla(f):
    def _nabla(_x, dx=0.1):
        dim = _x.shape[0]
        res = np.zeros(dim)
        _h = np.eye(dim) * dx * 0.5
        for _ in range(_x.shape[0]):
            res[_] = f(_x + _h[_]) - f(_x - _h[_])
        res *= 1 / dx
        return res
    return _nabla


def calc_distance(mask_data):
    if mask_data.ndim == 3:
        phi = (mask_data.sum(axis=2) > 0).astype(int)
    else:
        phi = (mask_data > 0).astype(int)
    phi[phi == 0] = -1
    return skfmm.distance(phi, dx=1.0)


def calc_travel(dist_field, pos, coef, threshold=0.0):
    if dist_field is None:
        return None
    mask = dist_field < threshold
    if mask[int(pos[0]), int(pos[1])]:
        return None
    phi = -np.ones_like(dist_field)
    phi[int(pos[0]), int(pos[1])] = 1
    phi = np.ma.MaskedArray(phi, mask)
    speed = np.exp(coef) ** dist_field
    trial = 0
    t_begin = time.time()
    while True:
        try:
            res = skfmm.travel_time(phi, speed, dx=1.0)
            return res
        except (RuntimeError, ValueError):
            print("\x1b[2KError in calc_travel()! trial={}".format(trial),
                  end="\r")
            speed += 0.1
            trial += 1
            if time.time() - t_begin > 0.5:
                print("\x1b[2KSolution not found! trial={}".format(trial))
                return None


def estimate_path(
        travel_field, p_start, p_target, step=1.0):
    '''
    Parameters:
    ----------
    travel_field, 2D traveling time field np.ndarray(shape=[height, width])
    p_start: coordinate of initial point, [numeric, numeric]
    p_target: coordinate of initial point, [numeric, numeric]
    coef: coefficient used in calculating the wave speed

    Returns:
    ----------
    is_converged: check if the solution converged to p_start
    path: solution, np.ndarray (shape=(lenght, 2))
    '''

    def _check_invalid(pos):
        height, width = travel_field.shape
        return not((0 <= pos[0] < height) and (0 <= pos[1] < width))

    if travel_field is None \
            or _check_invalid(p_start) \
            or _check_invalid(p_target):
        return False, np.array([])

    df = nabla(vector_field(travel_field))
    p_now = np.array(p_target)

    def fc(x):
        df_now = df(x, dx=step * 0.01)
        return -df_now / (np.linalg.norm(df_now) + 1e-10)

    path = []
    k2_pre = np.zeros(2)
    is_converged = True
    t_begin = time.time()
    while np.linalg.norm(p_now - p_start) > step:
        path.append(np.array(p_now))
        k1 = step * fc(p_now)
        k2 = step * fc(p_now + 0.5 * k1)
        k3 = step * fc(p_now + 0.5 * k2)
        k4 = step * fc(p_now + k3)
        p_now += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if np.linalg.norm(k2_pre + k2) < 1e-5 or time.time() > t_begin + 1.0:
            is_converged = False
            break
        k2_pre = k2
    if is_converged:
        path.append(np.array(p_now))
    return is_converged, np.array(path)[::-1]
