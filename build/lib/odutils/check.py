'''Utilities for checking data'''

import pandas as pd
import numpy as np

import warnings


def is_invalid_df(X, axis=0):
  '''Check whether DataFrame has invalid values'''

  cond_list = []
  labels = []

  labels.append('null')
  shape = X.shape
  if len(X.shape) == 3:
    is_null_arr = np.isnan(X)
    is_null = is_null_arr.any(axis=axis).reshape(-1, 1)
  elif len(X.shape) == 2:
    is_null_arr = np.isnan(X)
    is_null = is_null_arr.any(axis=axis)
  elif len(X.shape) == 1:
    is_null = is_null_arr = np.isnan(X)
  else:
    msg = f'X dimension should be 1 - 3. {len(X.shape)}  is specified.'
    raise ValueError(msg)

  cnt_null = is_null.sum()
  cond_list.append(is_null)

  labels.append('inf')
  if len(X.shape) == 3:
    is_inf_arr = np.isinf(X)
    is_inf = is_inf_arr.any(axis=axis).reshape(-1, 1)
  elif len(X.shape) == 2:
    is_inf_arr = np.isinf(X)
    is_inf = is_inf_arr.any(axis=axis)
  elif len(X.shape) == 1:
    is_inf = is_inf_arr = np.isinf(X)
  else:
    msg = f'X dimension should be 1 - 3 . {len(X.shape)}  is specified.'
    raise ValueError(msg)

  is_invalid_arr = np.logical_or(is_null_arr, is_inf_arr)

  cnt_inf = is_inf.sum()
  cond_list.append(is_inf)

  if isinstance(X, pd.DataFrame):
    cond_df = pd.concat(cond_list, axis=1)
    cond_df.columns = labels
  elif isinstance(X, np.ndarray):
    cond_df = np.concatenate(cond_list, axis=1)
    cond_df = pd.DataFrame(cond_df, columns=labels)

  return cond_df, is_invalid_arr


def get_invalid(X, axis=0, return_where=False):
  '''Get which columns/rows has invalid values'''

  cond_df, is_invalid_arr = is_invalid_df(X, axis)

  is_invalid = cond_df.any(axis=1)

  if axis == 0:
    if isinstance(X, pd.DataFrame):
      if return_where:
        return (X.columns[is_invalid], is_invalid_arr)
      else:
        return X.columns[is_invalid]
    else:
      if return_where:
        return (X.index[is_invalid], is_invalid_arr)
      else:
        return X.index[is_invalid]
  elif axis == 1:
    if return_where:
      return (X.index[is_invalid], is_invalid_arr)
    else:
      return X.index[is_invalid]
  else:
    msg = f'axis should be 0 or 1. {axis}  is specified.'
    raise ValueError(msg)


def check_X_invalid(X, logger=None, ori='row'):

  if ori == 'row':
    axis =  1
  elif ori == 'col':
    axis = 0
  else:
    msg = f'ori should be row or col. {ori}  is specified.'
    raise ValueError(msg)

  cond_df, is_invalid_arr = is_invalid_df(X, axis)

  cnt_null = cond_df['null'].sum()
  cnt_inf = cond_df['inf'].sum()

  emsg = 'X contains: '
  emsg += f'{cnt_null} NaNs, '
  emsg += f'{cnt_inf} infs'
  invalid_ut = cond_df.loc[is_invalid, :].index
  emsg += f'Invalid {ori}={invalid_ut}'

  if cnt_null + cnt_inf > 0:
    if logger:
      logger.warn(emsg)
    else:
      warnings.warn(emsg)

  return cond_df


def get_X_invalid_dict(X, return_where=True):
  invalids = get_invalid(X, axis=1, return_where=return_where)
  if return_where:
    invalid_inds, X_inv = invalids
  else:
    invalid_inds = invalids
  invalid_dict = {}
  for idx in invalid_inds:
    invalid_dict[idx] = list(get_invalid(X.loc[idx, :]))
  return invalid_dict, X_inv


