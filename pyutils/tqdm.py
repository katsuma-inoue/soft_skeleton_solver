#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys


def is_env_notebook():
    try:
        from IPython import get_ipython
        env_name = get_ipython().__class__.__name__
        if env_name == 'NoneType':
            return False
        if env_name == 'TerminalInteractiveShell':
            return False
    except ImportError:
        return False
    return True


if is_env_notebook():
    # Jupyter Notebook environment
    import tqdm.notebook
    from tqdm.notebook import *
    # sys.modules[__name__] = tqdm.notebook
else:
    import tqdm
    from tqdm import *
    # sys.modules[__name__] = tqdm
