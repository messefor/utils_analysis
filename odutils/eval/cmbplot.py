'''Plot combination eval figures.'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

import seaborn as sns
from matplotlib.gridspec import GridSpec

from ..metrics import calc_reg_metrics, calc_clf_metrics
from .eval import binalize
from .baseplot import (plot_ts_pred_true, plot_scatter,
                        put_metrics_text_on_anyax,
                        plot_roc, put_metrics_text_on_roc,
                        put_metrics_text_on_ax,
                        plot_res_hist, plot_cm, plot_qq,
                        plot_index_vs_error)


def plot_error_diag(res, prefix=None):
  '''Error diagnostics plot

  Parameter
  ------------
  res array-like residual
  prefix str
  '''
  prefix = '' if prefix is None else prefix

  fig, axes = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)

  # Error histogram
  ax = axes[0, 0]
  params = dict(bins=15, edgecolor='white', alpha=.7)
  _ = ax.hist(res, **params)
  ax.axvline(0, color='black', lw=1)
  ax.set_title('Error distribution')
  ax.set_title(prefix + ' ' + ax.get_title())

  # Error boxplot
  ax = axes[0, 1]
  ax.boxplot(res)
  ax.set_ylabel('Residuals(Error)')
  ax.set_title('Error distribution')
  ax.set_title(prefix + ' ' + ax.get_title())

  # Index vs error
  ax = axes[1, 0]
  fig, ax = plot_index_vs_error(res, ax=ax)
  ax.set_title(prefix + ' ' + ax.get_title())

  # QQ plot
  ax = axes[1, 1]
  fig, ax = plot_qq(res, ax=ax)
  ax.set_title('Q-Q Plot')
  ax.set_title(prefix + ' ' + ax.get_title())

  return fig, axes


def plot_eval_cmb(y_train, y_train_pred, y_val, y_val_pred, thresh,
                      x_val=None, return_metrics=False, config_plot=None):

  '''Plot evaluation figures of both train and validation dataset.

  Parameters
  ------------
  x_val : array-like, default=None
        Specifies x as the timeseries plot x-value.
        If not defined(None), timeindex is used as x value.

  return_metrics : boolean, default=False
        Whether to return evaluation metrics which calcucalted in the process.


  '''

  config_plot = {} if config_plot is None else config_plot

  y_val_bin = binalize(y_val, thresh)
  y_val_pred_bin = binalize(y_val_pred, thresh)

  y_train_bin = binalize(y_train, thresh)
  y_train_pred_bin = binalize(y_train_pred, thresh)

  nrows = 2
  ncols = 6
  grids = GridSpec(nrows=nrows, ncols=ncols)
  fig = plt.figure(figsize=(5 * ncols, 5 * nrows))

  # 時系列比較（評価）
  config = config_plot.get('ts_pred_true', {})
  ax = fig.add_subplot(grids[0, 0:2])
  fig, ax =\
    plot_ts_pred_true(y_val, y_val_pred, y_thresh=thresh,
                                              ax=ax, x=x_val, **config)
  ax.set_title('Val ' + ax.get_title())

  # 時系列比較（学習）
  config = config_plot.get('ts_pred_true', {})
  ax = fig.add_subplot(grids[1, 0:2])
  fig, ax =\
    plot_ts_pred_true(y_train, y_train_pred, y_thresh=thresh,
                                            x=None, ax=ax, **config)
  ax.set_title('Train ' + ax.get_title())

  # 誤差分布（評価）
  config = config_plot.get('res_hist', {})
  ax = fig.add_subplot(grids[0, 2])
  fig, ax = plot_res_hist(y_val, y_val_pred, ax=ax)
  metrics_reg_val = calc_reg_metrics(y_val, y_val_pred)
  ax = put_metrics_text_on_anyax(ax, x_ratio=0.7, y_ratio=0.9, **metrics_reg_val)

  # 誤差分布（学習）
  config = config_plot.get('res_hist', {})
  ax = fig.add_subplot(grids[1, 2])
  fig, ax = plot_res_hist(y_train, y_train_pred, ax=ax, **config)
  metrics_reg_tr = calc_reg_metrics(y_train, y_train_pred)
  ax = put_metrics_text_on_anyax(ax, x_ratio=0.7, y_ratio=0.9, **metrics_reg_tr)

  # 散布図（評価）
  config = config_plot.get('scatter', {})
  ax = fig.add_subplot(grids[0, 3])
  fig, ax = plot_scatter(y_val, y_val_pred, ax=ax, **config)
  metrics_reg_val = calc_reg_metrics(y_val, y_val_pred)
  ax = put_metrics_text_on_anyax(ax, x_ratio=0.2, y_ratio=0.9, **metrics_reg_val)

  # 散布図（学習）
  config = config_plot.get('scatter', {})
  ax = fig.add_subplot(grids[1, 3])
  fig, ax = plot_scatter(y_train, y_train_pred, ax=ax, **config)
  metrics_reg_tr = calc_reg_metrics(y_train, y_train_pred)
  ax = put_metrics_text_on_anyax(ax, x_ratio=0.2, y_ratio=0.9, **metrics_reg_tr)

  # 混合行列（評価）
  config = config_plot.get('cm', {})
  ax = fig.add_subplot(grids[0, 4])
  fig, ax = plot_cm(y_val_bin, y_val_pred_bin, ax=ax, **config)

  # 混合行列（学習）
  config = config_plot.get('cm', {})
  ax = fig.add_subplot(grids[1, 4])
  fig, ax = plot_cm(y_train_bin, y_train_pred_bin, ax=ax, **config)

  # ROC（評価）
  config = config_plot.get('roc', {})
  ax = fig.add_subplot(grids[0, 5])
  fig, ax, auc_val = plot_roc(y_val_bin, y_val_pred, ax=ax, return_auc=True)
  metrics_clf_val = calc_clf_metrics(y_val_bin, y_val_pred_bin)
  metrics_clf_val['auc'] = auc_val
  ax = put_metrics_text_on_roc(ax, metrics_clf_val)

  # ROC（学習）
  config = config_plot.get('roc', {})
  ax = fig.add_subplot(grids[1, 5])
  fig, ax, auc_train = plot_roc(y_train_bin, y_train_pred, ax=ax,
                            return_auc=True)
  metrics_clf_tr = calc_clf_metrics(y_train_bin, y_train_pred_bin)
  metrics_clf_tr['auc'] = auc_train
  ax = put_metrics_text_on_roc(ax, metrics_clf_tr)

  fig.tight_layout()

  if return_metrics:
    metrics_tr = metrics_clf_tr
    metrics_val = metrics_clf_val
    metrics_tr.update(metrics_reg_tr)
    metrics_val.update(metrics_reg_val)
    return fig, metrics_val, metrics_tr
  else:
    return fig


def plot_eval_cmb_test_only(y_test, y_test_pred, thresh, x=None,
                                      return_metrics=False, config_plot=None):
  '''Plot evaluation figures only for test dataset.

  Parameters:
  -------------------------
  x_val : array-like, default=None
        Specifies x as the timeseries plot x-value.
        If not defined(None), timeindex is used as x value.

  return_metrics : boolean, default=False
        Whether to return evaluation metrics which calcucalted in the process.

  '''

  config_plot = {} if config_plot is None else config_plot

  y_test_bin = binalize(y_test, thresh)
  y_test_pred_bin = binalize(y_test_pred, thresh)

  nrows = 1
  ncols = 6
  grids = GridSpec(nrows=nrows, ncols=ncols)
  fig = plt.figure(figsize=(5 * ncols, 5 * nrows))

  # 時系列比較（評価）
  # 学習データのend_timesは復数実験含んでいるため別途処理が必要。
  # なので学習データはtime_indexを使う
  config = config_plot.get('ts_pred_true', {})
  ax = fig.add_subplot(grids[0:2])
  fig, ax = plot_ts_pred_true(y_test, y_test_pred, y_thresh=thresh,
                              ax=ax, x=x, **config)

  # 誤差分布（評価）
  config = config_plot.get('res_hist', {})
  ax = fig.add_subplot(grids[2])
  fig, ax = plot_res_hist(y_test, y_test_pred, ax=ax, **config)
  metrics_reg = calc_reg_metrics(y_test, y_test_pred)
  ax = put_metrics_text_on_anyax(ax, x_ratio=0.7, y_ratio=0.9, **metrics_reg)

  # 散布図（評価）
  config = config_plot.get('scatter', {})
  ax = fig.add_subplot(grids[3])
  fig, ax = plot_scatter(y_test, y_test_pred, ax=ax, **config)
  metrics_reg = calc_reg_metrics(y_test, y_test_pred)
  ax = put_metrics_text_on_anyax(ax, x_ratio=0.2, y_ratio=0.9, **metrics_reg)

  # 混合行列（評価）
  config = config_plot.get('cm', {})
  ax = fig.add_subplot(grids[4])
  fig, ax = plot_cm(y_test_bin, y_test_pred_bin, ax=ax, **config)

  # ROC（評価）
  config = config_plot.get('roc', {})
  ax = fig.add_subplot(grids[5])
  fig, ax, auc_test = plot_roc(y_test_bin, y_test_pred, ax=ax, return_auc=True)
  metrics_clf = calc_clf_metrics(y_test_bin, y_test_pred_bin)
  metrics_clf['auc'] = auc_test
  ax = put_metrics_text_on_roc(ax, metrics_clf)

  fig.tight_layout()

  if return_metrics:
    metrics =  metrics_reg
    metrics.update(metrics_clf)
    return fig, metrics
  else:
    return fig


def plot_eval_cmb_total(y_train, y_train_pred, y_val, y_val_pred,
                              thresh, return_metrics=False, config_plot=None):
  '''No Timeseires plot and ROC curve.

  Parameters
  ----------
  x_val : array-like, default=None
        Specifies x as the timeseries plot x-value.
        If not defined(None), timeindex is used as x value.

  return_metrics : boolean, default=False
        Whether to return evaluation metrics which calcucalted in the process.


  '''

  config_plot = {} if config_plot is None else config_plot

  y_val_bin = binalize(y_val, thresh)
  y_val_pred_bin = binalize(y_val_pred, thresh)

  y_train_bin = binalize(y_train, thresh)
  y_train_pred_bin = binalize(y_train_pred, thresh)

  nrows = 2
  ncols = 4
  grids = GridSpec(nrows=nrows, ncols=ncols)
  fig = plt.figure(figsize=(5 * ncols, 5 * nrows))


  # 散布図（評価）
  config = config_plot.get('scatter', {})
  ax = fig.add_subplot(grids[0, 0])
  fig, ax = plot_scatter(y_val, y_val_pred, ax=ax, **config)

  # 散布図（学習）
  config = config_plot.get('scatter', {})
  ax = fig.add_subplot(grids[1, 0])
  fig, ax = plot_scatter(y_train, y_train_pred, ax=ax, **config)

  # 誤差分布（評価）
  config = config_plot.get('res_hist', {})
  ax = fig.add_subplot(grids[0, 1])
  fig, ax = plot_res_hist(y_val, y_val_pred, ax=ax, **config)
  metrics_reg_val = calc_reg_metrics(y_val, y_val_pred)
  ax = put_metrics_text_on_anyax(ax, x_ratio=0.7, y_ratio=0.9, **metrics_reg_val)

  # 誤差分布（学習）
  config = config_plot.get('res_hist', {})
  ax = fig.add_subplot(grids[1, 1])
  fig, ax = plot_res_hist(y_train, y_train_pred, ax=ax, **config)
  metrics_reg_tr = calc_reg_metrics(y_train, y_train_pred)
  ax = put_metrics_text_on_anyax(ax, x_ratio=0.7, y_ratio=0.9, **metrics_reg_tr)

  # 混合行列（評価）
  config = config_plot.get('cm', {})
  ax = fig.add_subplot(grids[0, 2])
  fig, ax = plot_cm(y_val_bin, y_val_pred_bin, ax=ax, **config)

  # 混合行列（学習）
  config = config_plot.get('cm', {})
  ax = fig.add_subplot(grids[1, 2])
  fig, ax = plot_cm(y_train_bin, y_train_pred_bin, ax=ax, **config)

  # 指標（評価）
  ax = fig.add_subplot(grids[0, 3])
  metrics_clf_val = calc_clf_metrics(y_val_bin, y_val_pred_bin)
  ax = put_metrics_text_on_ax(ax, metrics_clf_val)

  # 指標（学習）
  ax = fig.add_subplot(grids[1, 3])
  metrics_clf_tr = calc_clf_metrics(y_train_bin, y_train_pred_bin)
  ax = put_metrics_text_on_ax(ax, metrics_clf_tr)

  fig.tight_layout()

  if return_metrics:
    metrics_tr = metrics_clf_tr
    metrics_val = metrics_clf_val
    metrics_tr.update(metrics_reg_tr)
    metrics_val.update(metrics_reg_val)
    return fig, metrics_val, metrics_tr
  else:
    return fig


