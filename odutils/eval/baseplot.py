'''Base evaluation plots'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

import seaborn as sns

from sklearn.metrics import (confusion_matrix, roc_curve, auc )
from ..metrics import RMSE



# ------------------------------------------------------------------


def set_prop_ax(ax, **kwargs):

  argkeys =  kwargs.keys()

  if 'title' in argkeys:
    ax.set_title(kwargs['title'])

  if 'ylim' in argkeys:
    ax.set_ylim(*kwargs['ylim'])

  if 'xlim' in argkeys:
    ax.set_xlim(*kwargs['xlim'])

  if 'ylabel' in argkeys:
    ax.set_ylabel(kwargs['ylabel'])

  if 'xlabel' in argkeys:
    ax.set_xlabel(kwargs['xlabel'])

  return ax


def plot_cm(y_true, y_pred, ax=None, ax_kwargs=None, heatmap_kwargs=None):
  '''Plot comfusion matrix heatmap for evaluation.

    NOTE: x-axis: y_pred, y-axis: y_true are not consistent with plot_scatter()

  '''
  if ax is None:
    fig, ax = plt.subplots(figsize=(6, 5))
  else:
    fig = ax.get_figure()

  kwargs = dict(annot=True, fmt='.5g', cmap='Blues', cbar=False, ax=ax,
                  linewidths=0.1, linecolor='grey',annot_kws={'size': 18})
  heatmap_kwargs = {} if heatmap_kwargs is None else heatmap_kwargs
  kwargs.update(heatmap_kwargs)

  cm = confusion_matrix(y_true, y_pred)
  ax = sns.heatmap(cm, **kwargs)

  ax.set_ylabel('True', fontsize=12)
  ax.set_xlabel('Predicted', fontsize=12)
  title = 'Confusion Matrix'
  ax.set_title(title)

  ax_kwargs = {} if ax_kwargs is None else ax_kwargs
  ax = set_prop_ax(ax, **ax_kwargs)

  return fig, ax


def plot_scatter(y_true, y_pred, ax=None, lim=(-0.5, 4),
                    jitter=True, ax_kwargs=None, scatter_kwargs=None):
  '''Plot a y_true vs y_pred scatter figure for evaluation.

    x-axis: y_true, y-axis: y_pred

  '''

  if ax is None:
    fig, ax = plt.subplots(figsize=(5, 5))
  else:
    fig = ax.get_figure()


  cmp = plt.get_cmap('tab10')

  if jitter:
    eps = np.random.normal(0, 0.03, size=y_true.shape[0])

  kwargs = {'s': 20, 'color': cmp(0), 'alpha': 0.7}
  scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
  kwargs.update(scatter_kwargs)

  ax.scatter(y_true + eps, y_pred, **kwargs)

  ax.axhline(0, color='grey', lw=0.5)
  ax.axvline(0, color='grey', lw=0.5)

  if lim is not None:
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)

  ax.grid(True, ls='--', color='grey', lw=0.5)

  ax.set_xlabel('True', fontsize=12)
  ax.set_ylabel('Predicted', fontsize=12)

  title = 'scatter pred vs actual'
  ax.set_title(title)

  ax_kwargs = {} if ax_kwargs is None else ax_kwargs
  ax = set_prop_ax(ax, **ax_kwargs)

  vmax = max(ax.get_ylim()[1], ax.get_xlim()[1])
  ax.plot([0, vmax], [0, vmax], color='grey', lw=0.5)
  ax.set_xticks(np.arange(0, vmax + 1, 1))
  ax.set_yticks(np.arange(0, vmax + 1, 1))

  return fig, ax


def plot_ts_pred_true(y_true, y_pred, ax=None, x=None, y_thresh=None,
                                  ax_kwargs: dict=None, plot_kwargs=None):
  '''Plot a y_true / y_pred timeseries figure for evaluation.

  Parameters
  ------------
  x_val : array-like, default=None
        Specifies x as the timeseries plot x-value.
        If not defined(None), timeindex is used as x value.

  '''

  if ax is None:
    fig, ax = plt.subplots(figsize=(10, 5))
  else:
    fig = ax.get_figure()

  kwargs = dict(marker='o')
  plot_kwargs = {} if plot_kwargs is None else plot_kwargs
  kwargs.update(plot_kwargs)

  if x is None:
    ax.plot(y_true, label='True', **kwargs)
    ax.plot(y_pred, label='Predicted', **kwargs)
  else:
    ax.plot(x, y_true, label='True', **kwargs)
    ax.plot(x, y_pred, label='Predicted', **kwargs)

  title = 'Timeseries True/Pred'
  ax.set_title(title)

  ax_kwargs = {} if ax_kwargs is None else ax_kwargs
  ax = set_prop_ax(ax, **ax_kwargs)

  if y_thresh is not None:
    ax.axhline(y_thresh, color='r', lw=1, ls='--')

  ax.grid(color='grey', ls='--', lw=1.)

  ax.legend(loc='upper left')

  fig.tight_layout()

  return fig, ax


def plot_roc(y_true_bin, y_score, ax=None,
                    ax_kwargs=None, plot_kwargs=None, return_auc=False):
  '''Plot ROC curve plot from y_true, y_score'''
  fpr, tpr, _ = roc_curve(y_true_bin, y_score)
  auc_metrics = auc(fpr, tpr)
  fig, ax = plot_roc_base(fpr, tpr, ax=ax,
                      ax_kwargs=ax_kwargs, plot_kwargs=plot_kwargs)
  if return_auc:
    return fig, ax, auc_metrics
  else:
    return fig, ax


def plot_roc_base(fpr, tpr, ax=None, ax_kwargs=None, plot_kwargs=None):
  '''Plot ROC curve plot from fpr, tpr.


  '''

  if ax is None:
    fig, ax = plt.subplots(figsize=(10, 10))
  else:
    fig = ax.get_figure()

  kwargs = dict(lw=1.)
  plot_kwargs = {} if plot_kwargs is None else plot_kwargs
  kwargs.update(plot_kwargs)

  if fpr.ndim == 1:
    label = 'ROC Curve'
    label = '{} (area={:.3f})'.format(label, auc(fpr, tpr))
    ax.plot(fpr, tpr, label=label, **kwargs)
  else:
    for f, t, lbl in zip(fpr, tpr, label):
      l = '{} (area={:.2f})'.format(lbl, auc(f, t))
      ax.plot(fpr, tpr, label=l, **kwargs)

  ax.plot([0, 1], [0, 1], 'k--', lw=1.)
  ax.set_xlabel('False positive rate')
  ax.set_ylabel('True positive rate')
  ax.set_title('ROC curve')
  ax.legend(loc='lower right')

  ax_kwargs = {} if ax_kwargs is None else ax_kwargs
  ax = set_prop_ax(ax, **ax_kwargs)

  fig.tight_layout()

  return fig, ax


def plot_roc_multi(fpr, tpr, roc_auc, ax=None):
  '''Plot ROC curve with multiple legend on one ax.

  '''

  if ax is None:
    fig, ax = plt.subplots(figsize=(8, 8))
  else:
    fig = ax.get_figure()

  for (lbl, f), (_, t) in zip(fpr.items(), tpr.items()):
    kwargs = dict(lw=1.)
    l = '{} (area={:.2f})'.format(lbl, auc(f, t))
    if lbl == 'macro':
      kwargs.update({'ls': '--'})
    ax.plot(f, t, label=l, **kwargs)

  ax.plot([0, 1], [0, 1], 'k--', lw=1.)
  ax.set_xlabel('False positive rate')
  ax.set_ylabel('True positive rate')
  ax.set_title('ROC curve')
  ax.legend(loc='lower right')

  fig.tight_layout()

  return fig, ax


def put_metrics_text_on_roc(ax, metrics):
  d = 0.08
  i = 0
  for k, v in metrics.items():
    if k != 'neg_pos_cnt':
      ax.text(0.7, 0.4 - i * d, '{}={:.2f}'.format(k, v))
      i += 1
  return ax


def put_metrics_text_on_anyax(ax, x_ratio=0.3, y_ratio=0.7, **kwargs):
  n_dev = 20
  ymin, ymax = ax.get_ylim()
  xmin, xmax = ax.get_xlim()
  x_start = (1 - x_ratio) * xmin +  x_ratio * xmax
  y_start = (1 - y_ratio) * ymin + y_ratio * ymax
  d = (ymax - ymin) / n_dev
  metrics = kwargs
  i = 0
  for k, v in metrics.items():
    ax.text(x_start, y_start - i * d, '{}={:.2f}'.format(k, v))
    i += 1
  return ax


def put_metrics_text_on_ax(ax, metrics):
  ax.tick_params(color='white')
  ax.tick_params(labelbottom=False, labelleft=False,
                  labelright=False, labeltop=False)
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  d = 0.1
  i = 0
  for k, v in metrics.items():
    if k != 'neg_pos_cnt':
      ax.text(0.1, 0.7 - i * d, '{}={:.2f}'.format(k, v), size=18)
      i += 1
  return ax


def plot_res_hist(y_true, y_pred, ax=None, hist_kwargs=None, ax_kwargs=None):
  '''Plot residual histogram. '''

  if ax is None:
    fig, ax = plt.subplots(figsize=(6, 6))
  else:
    fig = ax.get_figure()

  kwargs = {'bins': int(np.sqrt(len(y_true))),
            'edgecolor': 'w', 'alpha': 0.7}
  hist_kwargs = {} if hist_kwargs is None else hist_kwargs
  kwargs.update(hist_kwargs)

  ax.hist(y_true - y_pred, **kwargs)

  ax.axvline(0, color='grey', lw=1., ls='--')

  ax.set_xlabel('Residual (true - pred)')
  ax.set_ylabel('Frequency')
  ax.set_title('Distribution of residuals')

  ax_kwargs = {} if ax_kwargs is None else ax_kwargs
  ax = set_prop_ax(ax, **ax_kwargs)

  fig.tight_layout()

  return fig, ax


# ------------------------------------------------------------------------
# Learning curve
# ------------------------------------------------------------------------

def build_learning_curve(X_train, y_train, X_val, y_val, estimator,
                      max_n_train=None, step=1, eval_func=RMSE, as_frame=True):
  max_n_train = X_train.shape[0] if max_n_train is None else max_n_train
  indices = range(max_n_train)

  metrices_list = []
  for size_fetch in np.arange(1, max_n_train + 1, step=step):
    ids = np.random.choice(indices, size=int(size_fetch))
    X_train_s = X_train.iloc[ids, :]
    y_train_s = y_train[ids]

    est = deepcopy(estimator)
    est.fit(X_train_s, y_train_s)

    metrices = {'size': size_fetch}
    y_train_pred = est.predict(X_train_s)
    mtrx_train = eval_func(y_train_s, y_train_pred)
    metrices['train'] = mtrx_train

    y_val_pred = est.predict(X_val)
    mtrx_val = eval_func(y_val, y_val_pred)
    metrices['val'] = mtrx_val

    metrices_list.append(metrices)

  if as_frame:
    return pd.DataFrame(metrices_list).sort_values(by='size'), est
  else:
    return metrices_list, est


def plot_learning_curve(lc_mtrx_org, ax=None, with_band=False, goal=0.5):

  if ax is None:
    fig, ax = plt.subplots(figsize=(12, 5))
  else:
    fig = ax.get_figure()

  lc_mtrx = lc_mtrx_org.copy()
  lc_mtrx['train_ma'] = lc_mtrx['train'].rolling(5).mean()
  lc_mtrx['val_ma'] = lc_mtrx['val'].rolling(5).mean()

  cmap = plt.get_cmap('tab10')

  if with_band:
    ax.plot(lc_mtrx['size'], lc_mtrx['train'], label='train',
                                                  alpha=1., color=cmap(0))
    ax.plot(lc_mtrx['size'], lc_mtrx['val'], label='val',
                                                  alpha=1., color=cmap(1))
    ax.fill_between(lc_mtrx['size'],
                    lc_mtrx['train'] - lc_mtrx['train_std'],
                    lc_mtrx['train'] + lc_mtrx['train_std'],
                    label='train std', alpha=.4, color=cmap(0))
    ax.fill_between(lc_mtrx['size'],
                    lc_mtrx['val'] - lc_mtrx['val_std'],
                    lc_mtrx['val'] + lc_mtrx['val_std'],
                    label='val std', alpha=.4, color=cmap(1))
  else:
    ax.plot(lc_mtrx['size'], lc_mtrx['train'], label='train', ls='--',
                                                  alpha=.4, color=cmap(0))
    ax.plot(lc_mtrx['size'], lc_mtrx['val'], label='val', ls='--',
                                                  alpha=.4, color=cmap(1))
    ax.plot(lc_mtrx['size'], lc_mtrx['train_ma'], label='train_ma',
                                                alpha=1., color=cmap(0))
    ax.plot(lc_mtrx['size'], lc_mtrx['val_ma'], label='val_ma',
                                                alpha=1., color=cmap(1))

  ax.axhline(goal, color='red', ls='--', label='Goal={}'.format(goal))
  title = 'Learning Curve'
  ax.set_title(title)
  ax.set_ylabel('Loss')
  ax.set_xlabel('# of training samples')
  ax.legend(loc='upper right')
  ax.grid(color='grey', lw=1., ls='--')

  return fig, ax


# -----------------------------------------------------------------
# importance
# -----------------------------------------------------------------


def extract_importance(est, feats_nm, max_feats_nm_len=20):
  if hasattr(est, 'feature_importances_'):
    score = est.feature_importances_
    imps = {'importance': score,
                  'feats_nm': feats_nm}
    imp = pd.DataFrame(imps).sort_values(by='importance', ascending=False)
  elif hasattr(est, 'coef_'):
    score = est.coef_
    imps = {'importance': score,
            'importance_abs': np.abs(score),
                  'feats_nm': feats_nm}
    imp = pd.DataFrame(imps).sort_values(by='importance_abs', ascending=False)
    imp = imp[['importance', 'feats_nm']]
  else:
    raise NotImplementedError
  imp['feats_nm_shorten'] = imp['feats_nm'].str.slice(0, max_feats_nm_len)
  return imp


def plot_importance(est, feats_nm, top_n=20, ax=None, max_feats_nm_len=20):

  if ax is None:
    fig, ax = plt.subplots(figsize=(5, top_n / 2))
  else:
    fig = ax.get_figure()

  try:
    imp = extract_importance(est, feats_nm, max_feats_nm_len)
  except NotImplementedError:
    return None, None
  else:

    imp_top = imp.head(n=top_n).reset_index()

    y = imp_top['feats_nm_shorten']
    w = imp_top['importance']
    ax.barh(range(len(w)), w, tick_label=y)

    for i, v in enumerate(w):
      ax.text(v, i, '{:.2f}'.format(v))

    ax.invert_yaxis()
    ax.grid(color='grey', axis='x', lw=0.5)

    title = 'Importance of features top{}'.format(top_n)
    ax.set_title(title)

    fig.tight_layout()

    return fig, ax


