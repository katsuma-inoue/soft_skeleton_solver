#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "calc_curvature", "eval_performance",
    "calc_memory_function", "calc_parity_function",
    "effective_dimension"
]

import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge


def calc_curvature(data, slide=1):
    norm1 = np.linalg.norm(
        data[:, slide:-slide] - data[:, :-2 * slide], axis=2)
    norm2 = np.linalg.norm(
        data[:, 2 * slide:] - data[:, slide:-slide], axis=2)
    norm3 = np.linalg.norm(
        data[:, 2 * slide:] - data[:, :-2 * slide], axis=2)
    cross = np.cross(
        data[:, 2 * slide:] - data[:, slide:-slide],
        data[:, :-2 * slide] - data[:, slide:-slide], axis=2)
    curvature = cross / (norm1 * norm2 * norm3 + 1e-3)
    return curvature


def eval_performance(X_train, Y_train, X_eval, Y_eval, model_type="logistic"):
    if model_type == "logistic":
        model = LogisticRegression(random_state=None, solver="lbfgs")
        model.fit(X_train, Y_train)
        Y_out = model.predict(X_eval)
        size = Y_out.shape[0]
        mi = 0.0
        for _p, _q in itertools.product(model.classes_, model.classes_):
            b_x = Y_out == _p
            b_y = Y_eval == _q
            p_x = b_x.sum() / size
            p_y = b_y.sum() / size
            p_xy = np.logical_and(b_x, b_y).sum() / size + 1e-10
            mi += p_xy * np.log2(p_xy / (p_x * p_y + 1e-10))
        return mi
    if model_type == "linear":
        model = LinearRegression(fit_intercept=False)
        # model.fit(X_train, Y_train)
        # return ((Y_eval - Y_out)**2).sum() / (Y_eval ** 2).sum()

        Y_t = np.array(Y_train) * 2 - 1
        model.fit(X_train, Y_t)
        print(X_train.shape, Y_t.shape)
        w_model = np.linalg.inv(X_train.T.dot(X_train))

        Y_out = model.predict(X_eval)
        Y_out = Y_out > 0
        # Y_e = Y_eval
        size = Y_out.shape[0]
        mi = 0.0
        for _p, _q in itertools.product([0, 1], [0, 1]):
            b_x = Y_out == _p
            b_y = Y_eval == _q
            p_x = b_x.sum() / size
            p_y = b_y.sum() / size
            p_xy = np.logical_and(b_x, b_y).sum() / size + 1e-10
            mi += p_xy * np.log2(p_xy / (p_x * p_y + 1e-10))
        return mi
        # return (Y_o == Y_e).mean()


def calc_memory_function(
        X, Y, time_list, train_eval_ratio=0.5, noise_amp=0, **kwargs):
    assert len(X) == len(Y), "shapes must be same."
    size = len(X)
    X_pert = X + noise_amp * np.random.randn(*X.shape)
    border_id = int(size * train_eval_ratio)
    X_train, X_eval = X_pert[:border_id], X_pert[border_id:]
    Y_train, Y_eval = Y[:border_id], Y[border_id:]
    tau_margin = max(np.abs(time_list))
    result = []
    print(time_list)
    for _t in time_list:
        slice_x_train = slice(tau_margin, len(X_train) - tau_margin)
        slice_x_eval = slice(tau_margin, len(X_eval) - tau_margin)
        slice_y_train = slice(tau_margin - _t, len(Y_train) - tau_margin - _t)
        slice_y_eval = slice(tau_margin - _t, len(Y_eval) - tau_margin - _t)
        _score = eval_performance(
            X_train[slice_x_train], Y_train[slice_y_train],
            X_eval[slice_x_eval], Y_eval[slice_y_eval], **kwargs)
        result.append(_score)
        print("step {}: {:.3f}".format(_t, result[-1]))
    return np.array(result)


def calc_parity_function(
        X, Y, time_list, train_eval_ratio=0.5, noise_amp=0, **kwargs):
    assert len(X) == len(Y), "shapes must be same."
    X_pert = X + noise_amp * np.random.randn(*X.shape)
    result = []
    print(time_list)
    X_now, Y_now = X_pert[1:], Y[:-1]
    for _t in time_list:
        conv = np.ones(_t, dtype=int)
        Y_conv = np.convolve(Y_now, conv, "full")[:len(Y_now)]
        Y_conv = Y_conv % 2
        border_id = int(len(X_now) * train_eval_ratio)
        X_train, X_eval = X_now[:border_id], X_now[border_id:]
        Y_train, Y_eval = Y_conv[:border_id], Y_conv[border_id:]
        _score = eval_performance(X_train, Y_train, X_eval, Y_eval, **kwargs)
        result.append(_score)
        print("step {}: {:.3f}".format(_t, result[-1]))
    return np.array(result)


def effective_dimension(X):
    eigs = np.linalg.eig(X.T.dot(X))[0]
    es = abs(eigs)
    es *= 1 / sum(es)
    return 1 / (sum(es ** 2))
