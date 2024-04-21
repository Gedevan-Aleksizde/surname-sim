#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
import pandas as pd
from . import data


def load_dataset()->pd.DataFrame:
    with pkg_resources.path(data, 'myoji.csv') as datfile:
        return pd.read_csv(datfile, low_memory=False)