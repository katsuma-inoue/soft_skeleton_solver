#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cycler
import itertools
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams['lines.markersize'] = 8.0
matplotlib.rcParams['lines.markeredgewidth'] = 0.0
sns.set(font_scale=1.5, font="Arial")
sns.set_palette("tab10")
sns.set_style("whitegrid", {'grid.linestyle': '--'})

marker_cycle = cycler.cycler(marker=('o', '^', 's', 'X', 'p', 'D', 'P', 'H'))
matplotlib.rcParams['axes.prop_cycle'] += marker_cycle
