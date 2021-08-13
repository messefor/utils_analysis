'''学習用モジュール'''


from copy import deepcopy

import numpy as np
import pandas as pd

import logging
import toml


from sklearn.model_selection import GroupKFold
from utils.eval import MetricsCalc



class CVTrainer:
  '''CVでの精度確認用のTrainer

  学習データとSplitは固定で、変数やpipeline(model)のみ変更して評価する場合に利用
  '''
  def __init__(self, X_train: pd.DataFrame,
              y_train: (pd.Series, np.ndarray),
              groups_train: (pd.Series, np.ndarray), x_vars=None,
              metrics_calc=MetricsCalc(), sample_weight=None):

    self.X_train = X_train
    self.y_train = y_train.values \
                    if isinstance(y_train, pd.Series) else y_train
    self.groups_train = groups_train.values \
                    if isinstance(groups_train, pd.Series) else groups_train

    n_splits = len(np.unique(groups_train))
    self.kfold = GroupKFold(n_splits=n_splits)

    self.x_vars = X_train.columns if x_vars is None else x_vars

    self.metrics_calc = metrics_calc

    self.result_pred = None

    self.model_retrained = None

    self.sample_weight = sample_weight

  @property
  def pipelines(self):
    return {p['group']: p['pipeline'] for p in self.result_pred}

  @property
  def mean_metrics(self):
    '''評価指標のCV平均をtrain, valごとに算出'''
    gp_mean = self.result_metrics.groupby('kind').mean()
    is_valid = gp_mean.columns != 'group'
    return gp_mean.loc[:, is_valid]

  def _get_x_vars_not_exists(self):
    '''x_varsの変数がすべてXのカラムにあるか'''
    do_exists = np.isin(self.x_vars, self.X_train.columns)
    return self.x_vars[~do_exists]

  def train_all(self, pipeline):
    '''Splitせずに全学習データで学習 '''
    fit_kws = {}
    pipeline_clone = deepcopy(pipeline)
    X_tr = self.X_train[self.x_vars]
    y_tr = self.y_train
    if self.sample_weight is not None:
      fit_kws['sample_weight'] = self.sample_weight
    pipeline_clone.fit(X_tr, y_tr, **self.fit_kws)
    return pipeline_clone

  def train(self, pipeline, retrain=False):
    '''CVで学習・評価'''

    fit_kws = {}
    self.result_pred = []
    data_iter_train =\
      self.kfold.split(self.X_train, self.y_train, self.groups_train)
    for train_index, val_index in data_iter_train:

      grp = int(np.unique(self.groups_train[val_index]))

      X_tr = self.X_train.iloc[train_index][self.x_vars]
      y_tr = self.y_train[train_index]
      X_val = self.X_train.iloc[val_index][self.x_vars]
      y_val = self.y_train[val_index]

      if self.sample_weight is not None:
        sample_weight = self.sample_weight[train_index]
        fit_kws['sample_weight'] = sample_weight

      pipeline_clone = deepcopy(pipeline)
      pipeline_clone.fit(X_tr, y_tr, **fit_kws)

      y_tr_pred = pipeline_clone.predict(X_tr)
      y_val_pred = pipeline_clone.predict(X_val)

      mtx_tr = self.metrics_calc(y_tr, y_tr_pred)
      mtx_val = self.metrics_calc(y_val, y_val_pred)

      pred_info = {}
      pred_info['group'] = grp
      pred_info['pipeline'] = pipeline_clone

      pred_info['y_tr_true'] = y_tr
      pred_info['y_tr_pred'] = y_tr_pred
      pred_info['y_val_true'] = y_val
      pred_info['y_val_pred'] = y_val_pred

      pred_info['mtx_tr'] = mtx_tr
      pred_info['mtx_val'] = mtx_val

      self.result_pred.append(pred_info)

    # 評価指標を整理
    self.result_metrics = self.build_metrics_frame(self.result_pred)

    if retrain:
      # 全データで再学習
      self.model_retrained = self.train_all(pipeline)


  @staticmethod
  def build_metrics_frame(result_pred: list) -> pd.DataFrame:
    '''学習時に溜め込んだ結果から、評価指標のデータフレームを生成'''

    mtx_list = []
    for pred_info in result_pred:

      mtx_tr = pred_info['mtx_tr'].copy()
      mtx_tr['group'] = pred_info['group']
      mtx_tr['kind'] = 'train'
      mtx_list.append(mtx_tr)

      mtx_val = pred_info['mtx_val'].copy()
      mtx_val['group'] = pred_info['group']
      mtx_val['kind'] = 'val'
      mtx_list.append(mtx_val)

    return pd.DataFrame(mtx_list)

