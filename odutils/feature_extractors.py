
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, mode
from scipy import signal
from sklearn.linear_model import LinearRegression



from abc import ABCMeta, abstractmethod

from .window import WindowGenerator
from .misc import concat_dict, update_kwargs

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
# Calculate basic statistical features
# ----------------------------------------------------------------

def mode_value(x):
  return mode(x).mode[0]

STATS_FUNCS = {'mean': np.mean, 'std': np.std,
                'median': np.median, 'mode': mode_value,
                'kurtosis': kurtosis, 'skew': skew}

class StatWindowFeatsExtractor(BaseWindowFeatsExtactor):
  '''Statistic Features '''

  def __init__(self, target_cols, funcs=STATS_FUNCS, null_value=np.nan):

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

StatsWFExtractor = StatWindowFeatsExtractor  # Ailias


# ----------------------------------------------------------------
# Calculate linear trend (slope in time window)
# ----------------------------------------------------------------

class LinearTrendWFExtractor(BaseWindowFeatsExtactor):
  '''Linear trend features


  Return
  ---------------------
  dict
    * corr_coef corrcoef with time index
  '''


  def __init__(self, target_cols, col_time=None, lr_kwargs=None):

    self.target_cols = target_cols
    self.col_time = col_time
    self.lr_kwarg = lr_kwargs

    self.suffix_corr = 'corr_coef'
    self.suffix_lr = 'lr_coef'

    default_kwargs = dict(fit_intercept=True,
                          normalize=False)

    self.kwargs = update_kwargs(lr_kwargs, default_kwargs)

  def transform(self, d: pd.DataFrame) -> dict:

    # Corrcoef
    # Both will be the same if time index is evenly sampled with no missings
    if self.col_time is None:
      x = np.arange(d.shape[0])
    else:
      x = d[self.col_time].values - min(d[self.col_time].values)

    Y = d[self.target_cols].values
    corrs = np.corrcoef(x, Y.T)[0, 1:]
    feats_corr = {a + '_' + self.suffix_corr: b
                  for a, b in zip(self.target_cols, corrs)}

    # LinearFit slope
    feats_coef = {}
    X = x.reshape(-1, 1)
    for col in self.target_cols:
      estimator = LinearRegression(**self.kwargs)
      estimator.fit(X, d[col].values)
      feats_coef[col + '_' + self.suffix_lr] = estimator.coef_[0]

    return concat_dict(feats_corr, feats_coef)


# ----------------------------------------------------------------
# Calculate peak count
# ----------------------------------------------------------------

def get_filter_coef(fs, cutoff, numtaps, a=1,
                  kwargs=dict(window='hann')):
  '''Calculate FIR Lowpass filter coefficents'''

  dt = 1 / fs
  fn = fs / 2
  Wp = cutoff / fn
  b = signal.firwin(numtaps, Wp, **kwargs)
  return b, a


class PeakCountWFExtractor(BaseWindowFeatsExtactor):
  '''Peak count features '''

  def __init__(self, target_col, peak_finder=signal.find_peaks,
              fs=None, cutoff=None, numtaps=None,
              firwin_kwargs=dict(window='hann')):
    '''

    Parameters
    ----------------------
    fs  : int, default=None
      sampling frequency. Use when applying lowpass filter
    cutoff  : float, default=None
      cutoff frequency of FIR low pass filter. Use when applying lowpass filter
    numtaps  : int, default=None
      number of tups of FIR low pass filter. Use when applying lowpass filter

    '''

    self.suffix = 'n_peaks'

    self.target_col = target_col

    if fs is not None:

      if (cutoff is None) or (numtaps is None):
        msg = 'If you apply lowpass filter fs, cutoff, numtaps should be passed.'
        raise ValueError(msg)

      self.apply_lp = True

      b, a = get_filter_coef(fs, cutoff, numtaps, a=1, kwargs=firwin_kwargs)
      self.b = b
      self.a = a

      self.fs =fs
      self.cutoff = cutoff
      self.numtaps = numtaps
      self.delay = int((self.numtaps-1)/2)

    else:
      self.apply_lp = False

    self.peak_finder = peak_finder

  def transform(self, d: pd.DataFrame) -> dict:

    feats = {}

    y = d[self.target_col].values

    if self.apply_lp:
      y_lp = signal.lfilter(self.b, self.a, y)
      y = np.concatenate([y_lp[self.delay:], np.repeat(np.nan, self.delay)])

    self.y = y

    self.peak_inds, _ = self.peak_finder(y)
    n_peaks = len(self.peak_inds)
    feats[self.target_col + '_' + self.suffix] = n_peaks

    return feats



