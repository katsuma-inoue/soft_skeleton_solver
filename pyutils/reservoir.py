#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["DESN", "LESN", "Linear", "Softmax"]

import os
import sys
import joblib
import itertools
import scipy as sp
import numpy as np
import scipy.optimize
from numpy.random import RandomState
from pyutils.stats import sample_cross_correlation, optimize_ridge_criteria
from sklearn.linear_model import Ridge, LogisticRegression
from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence


class ESN(object):
    def __init__(self, dim, g=1.0, scale=None, noise_gain=None,
                 x_init=None, activation=np.tanh, dtype=None,
                 normalize=True, seed=None, tunable=False, **kwargs):
        self.dim = dim
        self.g = g
        self.noise_gain = noise_gain
        self.dtype = dtype
        if x_init is None:
            self.x_init = np.zeros(dim, dtype=self.dtype)
        else:
            self.x_init = np.array(x_init, dtype=self.dtype)
            assert (self.dim,) == x_init.shape, "shape of x_init must be" \
                " ({},...) (matrix with shape {} was given.)".format(
                    self.dim, self.x_init.shape)
        # internal values
        self.x = np.array(self.x_init)
        self.f = activation
        self.rnd = RandomState(seed)
        while True:
            try:
                self.scale = 1.0 if scale is None else scale
                coeff = 1.0 / np.sqrt(dim * self.scale)
                self.w_net = self.rnd.randn(dim, dim) * coeff
                self.w_net = self.w_net.astype(self.dtype)
                w_con = np.full((dim * dim,), False)
                w_con[:int(dim * dim * self.scale)] = True
                self.rnd.shuffle(w_con)
                w_con = w_con.reshape((dim, dim))
                self.w_net = self.w_net * w_con
                if normalize and self.dim > 0:
                    spectral_radius = max(abs(sp.sparse.linalg.eigs(
                        self.w_net, return_eigenvectors=False,
                        k=2, which="LM")))
                    self.w_net = self.w_net / spectral_radius
                break
            except ArpackNoConvergence:
                continue
        # parameters for innate learning
        self.tunable = tunable
        if self.tunable:
            self.reset_rls(**kwargs)

    def reset_rls(self, mu=1.0, alpha=1.0, select=None):
        self.tunable = True
        self.mu, self.alpha = mu, alpha
        w_pre_id = [self.w_net[_].nonzero()[0] for _ in range(self.dim)]
        if select is None:
            self.w_pre = w_pre_id
        else:
            self.w_pre = [_w[select(_w)] for _w in w_pre_id]
        self.P = {
            _: np.eye(self.w_pre[_].size, dtype=self.dtype) / self.alpha
            for _ in range(self.dim)}

    def to_pickle(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, mode="wb") as f:
            joblib.dump(self, f, compress=True)

    def f_g(self, x=None):
        if x is None:
            x = self.x
        return self.f(self.g * x)

    def step(self):
        raise NotImplementedError

    def step_while(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @staticmethod
    def read_pickle(file_name):
        with open(file_name, mode="rb") as f:
            net = joblib.load(f)
        return net


class DESN(ESN):
    '''
    Discrete-time echo state network

    Attributes:
        dim (int): number of nodes
        g (float): coefficient of function
        scale (float): density of connections
        noise_gain (float or array): Gaussian noise amplitude
        x_init (np.ndarray): init state
        activation: activation function of the network
        dtype (type): type of state and matrixs
        normalize (bool): normalize weight matrix if true
        seed (int): seed for random values (default: None)
        tunable (bool): enable tunable mode for innate training
        mu (float): forgetting parameter for innate training
        alpha (float): regularization strength for innate training
    '''
    def __init__(self, dim, **kwargs):
        super(DESN, self).__init__(dim, **kwargs)
        self.bias = None

    def step(self, u_in=None):
        self.x = self.x.dot(self.w_net.T)
        if u_in is not None:
            self.x += u_in
        if self.bias is not None:
            self.x += self.bias
        self.x = self.f_g(self.x)
        if self.noise_gain is not None:
            self.x += self.noise_gain * np.random.randn(self.dim)

    def step_while(self, num_step, u_in=None, init_step=0, verbose=False):
        for _ in range(init_step, num_step):
            if verbose:
                print("\x1b[2Kt={:.2f}".format(_), end="\r")
            if callable(u_in):
                self.step(u_in(_))
            else:
                self.step(u_in)

    def reset(self, x_reset=None):
        if x_reset is None:
            self.x = np.array(self.x_init)
        else:
            self.x = np.array(x_reset)

    def jacobian(self, x=None, use_cache=True):
        if x is None:
            x = self.x
        if not(use_cache and hasattr(self, "J1_")):
            self.J1_ = (self.w_net.T * self.g).T
        J2 = (1 / (np.cosh(self.g * x.dot(self.w_net.T))**2))
        return np.multiply(self.J1_.T, J2).T

    def maximum_lyapunov(self, num_step, u_in=None, num_trial=100,
                         perturbation=1e-6):
        result = []
        x_init = np.array(self.x)

        def _norm_rand():
            _noise = np.random.randn(num_trial, self.dim)
            return (_noise.T / np.linalg.norm(_noise, axis=1)).T
        x_pre = np.zeros((num_trial + 1, self.dim))
        x_pre[0] = self.x
        x_pre[1:] = self.x + perturbation * _norm_rand()
        self.reset(x_pre)
        self.step_while(num_step, u_in=u_in)
        x_post = np.array(self.x)
        d_post = np.linalg.norm(x_post[1:] - x_post[0], axis=1)
        result = np.log(d_post / perturbation) / num_step
        self.reset(x_init)  # reset to initial state
        return result


class LESN(ESN):
    '''
    Leaky echo state network

    Attributes:
        dim (int): number of ESN nodes
        tau (float): time constant of ESN
        g (float): coefficient of function
        mode (string): function types {"normal", "reverse"}
        scale (float): density of connection matrix
        noise_gain (float or array): Gaussian noise amplitude
        x_init (np.ndarray): init state
        activation: activation function of the network
        dtype (type): type of state and matrixs
        normalize (bool): normalize weight matrix if true
        seed (int): seed for random values (default: None)
        tunable (bool): enable tunable mode for innate training
        mu (float): forgetting parameter for innate training
        alpha (float): regularization strength for innate training
    '''
    def __init__(self, dim, tau, mode="normal", **kwargs):
        super().__init__(dim, **kwargs)
        self.tau = tau if np.isscalar(tau) else np.array(tau)
        assert mode in ["normal", "reverse"], \
            "option ***mode*** should be 'normal' or 'reverse'."
        self.mode = mode

    def fix_point(self, u_in=0.0, dim=None):
        net_range = slice(dim)
        x_init = np.zeros(self.dim)[net_range]
        _g = self.g if np.isscalar(self.g) else self.g[net_range]

        def _eq_normal(x, u_in=u_in):
            _x_in = _g * x.dot(
                self.w_net[net_range, net_range].T) + u_in[net_range]
            return -x[net_range] + self.f(_x_in)

        def _eq_reverse(x, u_in=u_in):
            _x_in = self.f(_g * x).dot(
                self.w_net[net_range, net_range].T)
            return -x + _x_in + u_in[net_range]

        if self.mode == "normal":
            return scipy.optimize.fsolve(_eq_normal, x_init)
        elif self.mode == "reverse":
            return scipy.optimize.fsolve(_eq_reverse, x_init)

    def step(self, dt, u_in=None):
        x_diff = np.zeros(self.x.shape)
        if u_in is None:
            u_in = 0.0
        if self.mode == "normal":
            x_diff += -self.x + self.f_g(self.x.dot(self.w_net.T) + u_in)
        elif self.mode == "reverse":
            x_diff += -self.x + self.f_g(self.x).dot(self.w_net.T) + u_in
        self.x += (dt / self.tau) * x_diff
        if self.noise_gain is not None:
            self.x += np.sqrt(2.0 * dt * self.noise_gain) * \
                np.random.randn(self.dim)

    def step_while(self, dt, T, u_in=None, t_init=0.0,
                   save=False, verbose=False):
        _t = t_init
        record = []
        while _t < t_init + T:
            if save:
                record.append(self.x)
            if verbose:
                print("\x1b[2Kt={:.2f}".format(_t), end="\r")
            if callable(u_in):
                self.step(dt, u_in(_t))
            else:
                self.step(dt, u_in)
            _t += dt
        return np.array(record)

    def reset(self, x_reset=None):
        if x_reset is None:
            self.x = np.array(self.x_init)
        else:
            self.x = np.array(x_reset)

    def jacobian(self, dt, x=None, use_cache=True):
        assert self.mode == "normal", "mode should be normal"
        if x is None:
            x = self.x
        if not(use_cache and hasattr(self, "J1_") and hasattr(self, "J2_")):
            self.J1_ = (1 - dt / self.tau) * np.eye(self.dim)
            self.J2_ = np.multiply(self.w_net.T, self.g * (dt / self.tau)).T
        return self.J1_ + self.J2_ * (1 / (np.cosh(self.g * x)**2))

    def maximum_lyapunov(self, dt, T, u_in=None, num_trial=100,
                         perturbation=1e-6, zero_range=None):
        result = []
        x_init = np.array(self.x)

        def _norm_rand():
            _noise = np.random.randn(num_trial, self.dim)
            if zero_range is not None:
                _noise[:, zero_range] = 0.0
            return (_noise.T / np.linalg.norm(_noise, axis=1)).T
        x_pre = np.zeros((num_trial + 1, self.dim))
        x_pre[0] = x_init
        x_pre[1:] = x_init + perturbation * _norm_rand()
        d_pre = np.linalg.norm(x_pre[1:] - x_pre[0], axis=1)
        self.reset(x_pre)
        self.step_while(dt, T, u_in=u_in)
        x_post = np.array(self.x)
        d_post = np.linalg.norm(x_post[1:] - x_post[0], axis=1)
        result = np.log(d_post / perturbation) / T
        self.reset(x_init)  # reset to previous state
        return result

    def maximum_lyapunov_dict(self, dt, T_list, u_in=None, num_trial=100,
                              perturbation=1e-6, zero_range=None):
        x_init = np.array(self.x)

        def _norm_rand():
            _noise = np.random.randn(num_trial, self.dim)
            if zero_range is not None:
                _noise[:, zero_range] = 0.0
            return (_noise.T / np.linalg.norm(_noise, axis=1)).T
        x_pre = np.zeros((num_trial + 1, self.dim))
        x_pre[0] = x_init
        x_pre[1:] = x_init + perturbation * _norm_rand()
        d_pre = np.linalg.norm(x_pre[1:] - x_pre[0], axis=1)

        result = {}
        self.reset(x_pre)
        T_pre = 0
        for _T in T_list:
            self.step_while(dt, _T - T_pre, u_in=u_in)
            x_post = np.array(self.x)
            d_post = np.linalg.norm(x_post[1:] - x_post[0], axis=1)
            result[_T] = np.log(d_post / perturbation) / _T
            T_pre = _T
        self.reset(x_init)  # reset to previous state
        return result

    def lyapunov_exponents(self, dt, x_list, size=None):
        r_list = []
        dim = x_list.shape[1]
        if size is None:
            size = self.dim
        q_pre = np.eye(self.dim, size)
        for _i, _x in enumerate(itertools.chain(x_list, x_list[::-1])):
            print("lyap: t={}".format(_i), end="\r")
            j = self.jacobian(dt, _x)
            q, r = np.linalg.qr(j.dot(q_pre))
            r_list.append(np.diag(r))
            q_pre = q
        l_list = np.log(np.abs(r_list))
        return l_list.mean(axis=0)

    def innate(self, x_target, x_now=None, neuron_list=None):
        assert self.tunable, "option ***tunable*** must be set to True" \
            " when you call this function."
        x_target = np.array(x_target)
        if x_now is None:
            x_now = self.x
        assert x_now.shape == x_target.shape, \
            "target shape must be same with that of current states (x_now)."
        if neuron_list is None:
            neuron_list = range(self.dim)

        def _innate(_now, _target):
            error = _now[neuron_list] - _target[neuron_list]
            for _id, _e in zip(neuron_list, error):
                x = _now[self.w_pre[_id]]
                k = self.P[_id].dot(x)
                c = 1 / (self.mu + x.dot(k.T))
                self.P[_id] -= np.outer(k, c * k)
                self.P[_id] *= 1 / self.mu
                self.w_net[_id, self.w_pre[_id]] -= k * (c * _e)

        if self.mode == "normal":
            val_now = x_now
            val_target = x_target
        elif self.mode == "reverse":
            val_now = self.f_g(x_now)
            val_target = self.f_g(x_target)
        if val_now.ndim == 1:
            _innate(val_now, val_target)
        elif val_now.ndim >= 2:
            for _now, _target in zip(
                    val_now.reshape(-1, self.dim),
                    val_target.reshape(-1, self.dim)):
                _innate(_now, _target)

    @staticmethod
    def concatenate(net_list):
        dim = sum([_net.dim for _net in net_list])
        # concatenating tau
        tau = []
        for _net in net_list:
            _tau = _net.tau
            if np.isscalar(_tau):
                _tau = np.ones(_net.dim) * _net.tau
            tau.append(_tau)
        tau = np.concatenate(tau)
        # concatenating g
        g = []
        for _net in net_list:
            _g = _net.g
            if np.isscalar(_g):
                _g = np.ones(_net.dim) * _net.g
            g.append(_g)
        g = np.concatenate(g)
        # creating new concatenated network
        net = LESN(dim, tau, g=g)
        net.w_net = np.zeros((dim, dim))
        dim_offset = 0
        for _net in net_list:
            dim_term = dim_offset + _net.dim
            net.w_net[dim_offset:dim_term, dim_offset:dim_term] = _net.w_net
            dim_offset = dim_term
        return net


class Readout(object):
    def __init__(self, dim_in, dim_out, dtype=None, w_init=None,
                 seed=None, distribution="uniform", dist_args={},
                 mu=1.0, alpha=1.0):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dtype = dtype
        self.rnd = RandomState(seed)
        if w_init is not None:
            self.w_init = w_init
            self.w_init = self.w.reshape((self.dim_out, self.dim_in))
        else:
            if hasattr(self.rnd, distribution):
                self.w_init = getattr(self.rnd, distribution)(
                    size=(self.dim_out, self.dim_in), **dist_args)
            elif hasattr(np, distribution):
                self.w_init = getattr(np, distribution)(
                    (self.dim_out, self.dim_in), **dist_args)
        self.w_init = self.w_init.astype(self.dtype)
        self.bias = np.zeros(self.dim_out)
        self.alpha = alpha
        self.mu = mu
        self.reset()

    def __call__(self, x_list, **kwargs):
        return self.predict(x_list, **kwargs)

    def reset(self):
        self.P = []
        for _ in range(self.dim_out):
            self.P.append(
                np.eye(self.dim_in, dtype=self.dtype) * (1 / self.alpha))
        self.w = np.array(self.w_init)

    def ridge(self, X, Y, alpha=1e-8, fit_intercept=True, **kwargs):
        if alpha == "auto":
            alpha = optimize_ridge_criteria(X, Y, creteria="AIC")
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        model.fit(X, Y)
        self.w, self.bias = model.coef_, model.intercept_

    def force(self, x, y):
        assert x.shape == (self.dim_in,), "input shape should be (dim_in,)"
        e = self.predict(x) - y
        dws = np.zeros(self.dim_in, self.dim_out)
        for _i in range(self.dim_out):
            k = self.P[_i].dot(x)
            c = 1 / (self.mu + x.dot(k))
            dP = np.outer(c * k, k)
            dw = (c * e[_i]) * k
            self.P[_i] -= dP
            self.P[_i] *= 1 / self.mu
            self.w[_i] -= dw
            dws[_i] = dw
        return dws

    def to_pickle(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, mode="wb") as f:
            joblib.dump(self, f, compress=True)

    @staticmethod
    def read_pickle(file_name):
        with open(file_name, mode="rb") as f:
            out = joblib.load(f)
        return out

    def predict(self, _input, **kwargs):
        raise NotImplementedError


class Linear(Readout):
    '''
    Linear tunable readout model

    Args:
        dim_in (int): input node dim.
        dim_out (int): output node dim.
        dtype (type, optional): numpy data type. Defaults to None.
        w_init (np.ndarray, optional): initial weight. Defaults to None.
        seed (int, optional): random seed. Defaults to None.
        distribution (str, optional): distribution type.
        dist_args (dict, optional): keyword args for np/rnd.
        mu (float, optional): forgetting parameter for innate training
        alpha (float, optional): regularization strength for innate training
    '''
    def predict(self, x):
        return np.array(x).dot(self.w.T) + self.bias


class Softmax(LogisticRegression):
    '''
    Alias of LogisticRegression
    '''
    pass
