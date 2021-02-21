#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["basename", "basename_without_ext",
           "dirname", "getdirs", "remove_ext", "extract_name"]

import os


def basename(path):
    '''
    alias of os.path.basename
    '''
    return os.path.basename(path)


def basename_without_ext(path):
    '''
    get basename without extension
    '''
    file_name = os.path.splitext(path)[0]
    return os.path.basename(file_name)


def dirname(path):
    '''
    alias of os.path.dirname
    '''
    return os.path.dirname(path)


def getdirs(path, func=None):
    '''
    get directory name list
    '''
    dirs = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            dirs.append(item)
    return sorted(dirs, key=func)


def remove_ext(path):
    '''
    remove extension from path
    '''
    return os.path.splitext(path)[0]


def extract_name(path):
    '''
    extract file name from path

    e.g.,
    a/b/c -> c
    a/b/c/ -> c
    a/b/c// -> c
    '''
    return os.path.basename(os.path.normpath(path))
