'''Metrics

'''

import numpy as np
import pandas as pd

from sklearn.metrics import (mean_squared_error, r2_score,
                              precision_recall_fscore_support,
                              accuracy_score, roc_curve, auc)
from .misc import cnv2ndarray


def RMSE(y_true, y_pred):
  return np.sqrt(mean_squared_error(y_true, y_pred))


def calc_mean_roc(pred_resuts):

  fpr, tpr, roc_auc = {}, {}, {}
  for k, vs in pred_resuts.items():
      y_true = vs['y_val_bin']
      y_score = vs['y_val_pred']
      fpr[k], tpr[k], _ = roc_curve(y_true, y_score)
      roc_auc[k] = auc(fpr[k], tpr[k])

  all_fpr = np.unique(np.concatenate(list(fpr.values())))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for k in pred_resuts.keys():
      mean_tpr += np.interp(all_fpr, fpr[k], tpr[k])
  mean_tpr /= len(pred_resuts)

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  return fpr, tpr, roc_auc


def calc_clf_metrics(y_true: (np.ndarray, pd.Series),
                     y_pred: (np.ndarray, pd.Series), labels=[0, 1]) -> dict:
  '''Calculate classification metrics.'''
  y_true = cnv2ndarray(y_true)
  y_pred = cnv2ndarray(y_pred)
  mss = precision_recall_fscore_support(y_true, y_pred, labels=labels)
  result = dict(accuracy=accuracy_score(y_true, y_pred),
                precision=mss[0][1],
                recall=mss[1][1],
                fscore=mss[2][1],
                neg_pos_cnt=mss[3],
                pos_ratio=mss[3][1] / mss[3].sum())
  return result


def calc_reg_metrics(y_true: (np.ndarray, pd.Series),
                    y_pred: (np.ndarray, pd.Series)) -> dict:
  '''Calculate regression metrics.'''
  y_true = cnv2ndarray(y_true)
  y_pred = cnv2ndarray(y_pred)
  result = dict(rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
                r2_score=r2_score(y_true, y_pred),
                corr=np.corrcoef(y_true, y_pred)[0, 1],
                )
  return result


def calc_all_metrics(y_true: (np.ndarray, pd.Series),
                    y_pred: (np.ndarray, pd.Series),
                    thresh: float=2.0) -> dict:
  '''Calculate metrics both for Regression and for Classification.'''
  metrics = {}

  mtc_reg = calc_reg_metrics(y_true, y_pred)
  metrics.update(mtc_reg)

  y_true_bin = (y_true >= thresh).astype(int)
  y_pred_bin = (y_pred >= thresh).astype(int)
  mtc_clf = calc_clf_metrics(y_true_bin, y_pred_bin, labels=[0, 1])
  metrics.update(mtc_clf)

  y_score = y_pred
  fpr, tpr, _ = roc_curve(y_true_bin, y_score)
  roc_auc = auc(fpr, tpr)
  metrics['auc'] = roc_auc

  return metrics


class MetricsCalculator:
  '''Class to calculate metrics'''

  def __init__(self, thresh=2.0):
    self.thresh = thresh

  def calc(self, y_true: (np.ndarray, pd.Series),
                y_pred: (np.ndarray, pd.Series)) -> dict:
    return calc_all_metrics(y_true, y_pred, thresh=self.thresh)

  def __call__(self, y_true: (np.ndarray, pd.Series),
                      y_pred: (np.ndarray, pd.Series)) -> dict:
    return self.calc(y_true, y_pred)

