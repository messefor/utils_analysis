import os

import numpy as np
import pandas as pd
import warnings


class EmptyWindowGeneratedError(ValueError):
  pass

class WindowGenerator:
  '''Generator fow windowing data'''
  def __init__(self, window_size: (int, float)=60,
                col_time: str='unixtime', min_length: int=100,
                edge_include: str='right', dropna=False,
                debug_savedir=None, raise_exc_when_small=False):

    self.window_size = window_size
    self.col_time = col_time
    self.min_length = min_length
    self.edge_include = edge_include
    self.dropna = dropna
    self.counter = 0
    self.init_yield = None
    self.debug_savedir = debug_savedir
    self.raise_exc_when_small = raise_exc_when_small


  def generate_err_msg(self):
    # FIXME: Logging
    emsg =  "=" * 30 + ".\n"
    emsg += "Too few rows exist in the window sliced.\n"
    emsg += 'dat_win.shape[0]: {}\n'.format(self.dat_win.shape[0])
    emsg += 'self.min_length: {}\n'.format(self.min_length)
    emsg += 'start_time, end_time: {}, {}\n'.format(self.start_time,
                                                    self.end_time)
    emsg += "=" * 30
    return emsg

  def generate(self, end_times: pd.Series, dat: pd.DataFrame):

    self.missing_data_end_times = []

    dat_s = dat.sort_values(by=self.col_time)

    for i, end_time in enumerate(end_times):

      start_time = end_time - self.window_size

      # Slice window
      # Chose which edge to include
      if self.edge_include == 'right':
        isin_win = (start_time < dat_s[self.col_time]) &\
                        (dat_s[self.col_time] <= end_time)
      elif self.edge_include == 'left':
        isin_win = (start_time <= dat_s[self.col_time]) &\
                        (dat_s[self.col_time] < end_time)
      elif self.edge_include == 'both':
        isin_win = (start_time <= dat_s[self.col_time]) &\
                        (dat_s[self.col_time] <= end_time)

      if self.dropna:
        dat_win = dat_s.loc[isin_win, :].dropna().reset_index(drop=True)
      else:
        dat_win = dat_s.loc[isin_win, :].reset_index(drop=True)

      self.start_time = start_time
      self.end_time = end_time
      self.dat_win = dat_win

      if (dat_win.shape[0] < self.min_length):

        self.missing_data_end_times.append(end_time)

        emsg = self.generate_err_msg()
        if self.raise_exc_when_small:
          raise EmptyWindowGeneratedError(emsg)
        else:
          warnings.warn(emsg + ' -> Returned: None')
          yield start_time, end_time, None

      else:

        if self.debug_savedir is not None:
          save_path = os.path.join(self.debug_savedir,
                                  f'window_{end_time}.csv')
          dat_win.to_csv(save_path, index=False)
          msg = '{}: Saving all windows data in to csv file.\n -> {}'.format(self.__class__.__name__, save_path)
          warnings.warn(msg)

        # Store initial value
        self.counter = i
        if self.counter == 0:
          self.init_yield = start_time, end_time, dat_win

        yield start_time, end_time, dat_win
