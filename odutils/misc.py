'''Misc utilities'''

import os
import pickle

import inspect
import logging

import numpy as np
import pandas as pd


def concat_dict(*args):
  '''Concatenate dictionary'''

  if len(args) == 0:
    return {}
  elif len(args) == 1:
    return args[0]
  if len(args) > 1:
    d = args[0]
    for a in args[1:]:
      d.update(a)
    return d
  else:
    return {}

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


def isinstance_all(arr, dtype):
  '''Is all elements in array same dtype?'''
  return np.all([isinstance(f, dtype) for f in arr])


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


def update_kwargs(kwargs=None, kwargs_default=None):
  '''Update keyword arguments '''
  kwargs_default = {} if kwargs_default is None else kwargs_default
  kwargs_out = kwargs_default.copy()
  kwargs_update = {} if kwargs is None else kwargs
  kwargs_out.update(kwargs_update)
  return kwargs_out

if __name__ == '__main__':

  # test update_kwargs()
  assert {} == update_kwargs()
  assert {'a':1, 'b':2, 'c': 5} == \
    update_kwargs(kwargs={'a': 1}, kwargs_default={'b':2, 'c': 5})
  assert {'a':1} ==  update_kwargs(kwargs={'a': 1})
  assert {'a':1, 'b':3, 'c': 5} == \
    update_kwargs(kwargs={'a': 1, 'b': 3}, kwargs_default={'b':2, 'c': 5})


  # test concat_dict()
  assert {} == concat_dict()
  assert {} == concat_dict({})
  assert {'a': 1} == concat_dict({'a': 1})
  assert {'a': 1, 'b': 2} == concat_dict({'a': 1}, {'b': 2})
  assert {'a': 1, 'b': 2, 'c': 3} ==\
    concat_dict({'a': 1}, {'b': 2}, {'c': 3})
  assert {'a': 2, 'b': 2} == concat_dict({'a': 1}, {'b': 2}, {'a': 2})

