
import time
import numpy as np
import pandas as pd
import warnings

from abc import ABCMeta, abstractmethod

from .window import WindowGenerator


# ---------------------------------------------------------------------------


class BaseWindowFeatsExtactor(metaclass=ABCMeta):

  def fit(self, d, **kwargs):
    return self

  @abstractmethod
  def transform(self, d: pd.DataFrame) -> (dict, np.ndarray, tuple):
    '''When transform() returns np.ndarray or tuple,
      we should specify column names as instance variables'''
    pass

  def transform_generator(self, gen, **kwargs) -> pd.DataFrame:
    '''Calculate feature from generator'''
    feats_list = []
    end_times = []
    for start_time, end_time, d in gen:
      self.end_time = end_time
      feat = self.transform(d, **kwargs)
      feats_list.append(feat)
      end_times.append(end_time)
    feats_df = self.list2frame(feats_list)
    feats_df.index = end_times
    feats_df.index.name = 'end_time'
    return feats_df

  def transform_generator_group(self, window_gen, end_time_group_gen):
    '''Calculate feature from generator with using end_time_generator

        You can use when you want apply transform by each group
    '''
    feats_list = []
    self.missing_data_end_times = {}
    index_by_keys = True
    for keys, end_times, d in end_time_group_gen:
      if index_by_keys:
        n = len(end_times)
        miarr = np.array([np.repeat(k, n) for k in keys] + [end_times])
      gen = window_gen.generate(end_times, d)
      feats_df = self.transform_generator(gen)
      if len(window_gen.missing_data_end_times) > 0:
        keys_h = self.key2hashable(keys)
        self.missing_data_end_times[keys_h] = window_gen.missing_data_end_times
      if index_by_keys:
        feats_df.index = pd.MultiIndex.from_arrays(miarr)
      feats_list.append(feats_df)
    feats = pd.concat(feats_list, axis=0)
    return feats

  @staticmethod
  def key2hashable(keys):
    if isinstance(keys, str):
      return keys
    else:
      if len(keys) == 1:
        return keys[0]
      else:
        return tuple(keys)

  @staticmethod
  def list2frame(feats_list):
    '''Convert list to DataFrame'''
    first_feat = feats_list[0]
    if isinstance(first_feat, pd.DataFrame):
      feats_df = pd.concat(feats, axis=0)
    elif isinstance(first_feat, np.ndarray):
      feats_arr = np.concatenate(feats_list, axis=0)
      return feats_arr
    elif isinstance(first_feat, dict):
      feats_df = pd.DataFrame(feats_list)
    else:
      msg = 'self.transform() should return dict or pd.DataFrame or np.ndarray type value. got type:{}'.format(type(first_feat))
      raise ValueError(msg)
    return feats_df




# ----------------------------------------------------------------

class StatWindowFeatsExtractor(BaseWindowFeatsExtactor):
  '''Statistic Features '''

  def __init__(self, target_cols,
              funcs={'mean': np.mean, 'std': np.std, 'median': np.median},
              null_value=np.nan):

    self.target_cols = target_cols
    self.funcs = funcs

    self.null_value = null_value
    self.value_missings ={col + '_' + k: null_value
                      for col in self.target_cols
                        for k, f in self.funcs.items()}

  def transform(self, d: pd.DataFrame) -> dict:

    if d is None:
      return self.value_missings
    else:
      return {col + '_' + k: f(d[col].values)
                    for col in self.target_cols
                      for k, f in self.funcs.items()}



