'''Evaluation module'''

import numpy as np
import pandas as pd

from ..metrics import calc_clf_metrics


def binalize(y, thresh):
    return (y >= thresh).astype(int)


def build_pr_table(y_test_pred, y_test_bin):
  '''Make Precision/Recall tradeoff table'''

  # Loop each treshold and calc metrics
  pr_list = []
  y_test_pred_s = np.sort(y_test_pred)
  for thresh in y_test_pred_s:
    y_test_pred_bin_tmp = (y_test_pred >= thresh).astype(int)
    r = calc_clf_metrics(y_test_bin, y_test_pred_bin_tmp)
    r['thresh'] = thresh
    pr_list.append(r)
  pr_df = pd.DataFrame(pr_list)

  # Add thresh where fscore takes its max
  pr_df['fscore_max'] = False
  pr_df.loc[pr_df['fscore'].idxmax(), 'fscore_max'] = True

  return pr_df


def get_optim_PR(pr_table: pd.DataFrame,
                      thresh_lb=None, thresh_ub=None):
  '''Calculate optimum Precision/Recall in PR table '''

  is_target = np.repeat(True, pr_table.shape[0])

  thresh_lb = np.nan if thresh_lb is None else thresh_lb
  thresh_ub = np.nan if thresh_ub is None else thresh_ub

  if not np.isnan(thresh_lb):
    is_target = is_target & (pr_table['thresh'] >= thresh_lb)

  if not np.isnan(thresh_ub):
    is_target = is_target & (pr_table['thresh'] < thresh_ub)

  idx_max_inarea = pr_table.loc[is_target, 'fscore'].idxmax()
  thesh_opt, prec_opt, recall_opt =\
    pr_table.loc[idx_max_inarea, ['thresh', 'precision', 'recall']]

  return thesh_opt, prec_opt, recall_opt