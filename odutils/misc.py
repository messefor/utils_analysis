'''Misc utilities'''

import os
import pickle

import inspect
import logging

import numpy as np
import pandas as pd


def isinstance_all(arr, dtype):
  '''Is all elements in array same dtype?'''
  return np.all([isinstance(f, dtype) for f in arr])


def save_pickle(x, pklpath):
  with open(pklpath, 'wb') as f:
    pickle.dump(x, f)


def load_pickle(pklpath):
  with open(pklpath, 'rb') as f:
    return pickle.load(f)


def makedir_p(data_file_path):
  '''Makedir w.r.t fpath'''
  dirpath = os.path.dirname(data_file_path)
  os.makedirs(dirpath, exist_ok=True)
  return dirpath


def is_run_by_kernel():
    '''Check if Python is run by interactive kernel.'''
    for f in inspect.stack():
        if f.filename.endswith('interactiveshell.py'):
            return True
    return False


def cnv2ndarray(x) -> np.ndarray:
  '''Convert variable into np.ndarray'''
  if isinstance(x, np.ndarray):
    return x
  elif isinstance(x, (pd.DataFrame, pd.Series)):
    return x.values
  elif isinstance(x, (list, tuple, set)):
    return np.ndarray(x)
  else:
    type_covered = ['pd.DataFrame', 'pd.Series',
                    'pd.ndarray', 'list', 'tuple', 'set']
    errmsg = 'Argument x should be one of {}'.format(type_covered)
    raise ValueError(errmsg)