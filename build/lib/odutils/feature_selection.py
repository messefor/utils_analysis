
import numpy as np
import pandas as pd

import warnings

from sklearn.feature_selection import mutual_info_regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from .misc import cnv2ndarray
from .train import  CVTrainer

def corrcoef_Xy(X: np.ndarray, y: np.ndarray) -> np.ndarray:
  '''Calculate corrcoef between X and y'''
  X, y = cnv2ndarray(X), cnv2ndarray(y)
  y_c = y - y.mean()
  X_c = X - X.mean(axis=0)
  N = y_c.shape[0]
  return (y_c @ X_c) / (N * y_c.std() * X_c.std(axis=0))


def check_has_variance(X: pd.DataFrame,
                              thresh_std=0, max_bin_ratio=0.95):
  '''Calculate variance of each variable'''

  conds = []

  # Calculate std
  std = X.std(axis=0)
  conds.append(std > thresh_std)

  def get_bin_ratio_max(x):
    '''最も多いカテゴリの比率を計算'''
    nrows = len(x)
    return np.max(np.unique(x, return_counts=True)[1] / nrows)

  # Calculate max bin frequnency ratio
  bin_ratio = X.apply(get_bin_ratio_max, axis=0)
  conds.append(bin_ratio <= max_bin_ratio)

  has_variance = np.array(conds).all(axis=0)

  return has_variance



def filter_useless_feats(X, y, thresh_std=0, **kwargs):

  msg_tmp = 'In X.shape[1] == {}, {} features has NO {}.'

  # 分散がないものを除外
  n_feats_before = X.shape[1]
  std = X.std(axis=0)
  has_variance = std > thresh_std
  X = X.loc[:, has_variance]  # 除外
  print(msg_tmp.format(n_feats_before, sum(~has_variance), 'variance'))

  # ほとんど同じ値の変数を除外
  if 'max_bin_ratio' in kwargs.keys():

    # Calculate max bin frequnency ratio
    def get_bin_ratio_max(x):
      '''最も多いカテゴリの比率を計算'''
      nrows = len(x)
      return np.max(np.unique(x, return_counts=True)[1] / nrows)

    n_feats_before = X.shape[1]
    bin_ratio = X.apply(get_bin_ratio_max, axis=0)
    has_divers = bin_ratio <= kwargs['max_bin_ratio']
    X = X.loc[:, has_divers]
    print(msg_tmp.format(n_feats_before, sum(~has_divers), 'diversity'))

  # 相関がないものを除外
  if 'thresh_corrcoef' in kwargs.keys():
    n_feats_before = X.shape[1]
    corrcoef = corrcoef_Xy(X, y)
    has_corrcoef = np.abs(corrcoef) > kwargs['thresh_corrcoef']
    X = X.loc[:, has_corrcoef]
    print(msg_tmp.format(n_feats_before, sum(~has_corrcoef), 'corrcoef'))

  # 相互情報量がないものを除外
  if 'thresh_mutual_info' in kwargs.keys():
    n_feats_before = X.shape[1]
    mi = mutual_info_regression(X, y)
    has_mutual_info = mi > kwargs['thresh_mutual_info']
    X = X.loc[:, has_mutual_info]
    print(msg_tmp.format(n_feats_before, sum(~has_mutual_info), 'mutual info'))

  return X



def calc_univar_model_metrics(estimator,
                              X, y, groups, x_vars_cands=None,
                                              max_n_feats=100):
  '''候補変数セットの中の1変数モデルの精度をCVで確認する'''
  x_vars_cands = X.columns if x_vars_cands is None else x_vars_cands
  max_n_feats = len(x_vars_cands) if max_n_feats is None else max_n_feats
  max_n_feats = min(max_n_feats, len(x_vars_cands))

  cvt = CVTrainer(X, y, groups)

  result = []
  for i in range(max_n_feats):
    cvt.x_vars = [x_vars_cands[i]]
    cvt.train(estimator, retrain=False)
    mms = cvt.mean_metrics
    mms['x_var'] = x_vars_cands[i]
    result.append(mms)
  result_df = pd.concat(result).reset_index()

  return result_df


def build_importance_frame(x_vars, importances):
  '''Format feature importance dataframe'''
  imp_info = dict(x_var=x_vars,
                    importance=importances)
  imp_info = pd.DataFrame(imp_info)
  imp_info = imp_info.sort_values(by='importance', ascending=False)
  imp_info = imp_info.reset_index(drop=True)
  imp_info['rank'] = range(1, imp_info.shape[0] + 1)
  return imp_info


def extract_cvt_importance(cvt):
  '''cvtのpipelineから変数のimportanceを取得する'''
  imporance_list = []
  for grp, pipline in cvt.pipelines.items():
    estimator = pipline
    x_vars = cvt.x_vars
    imps = extract_importance(estimator)
    importance_df = build_importance_frame(x_vars, imps)
    importance_df['group'] = grp
    imporance_list.append(importance_df)
  imps = pd.concat(imporance_list)
  return imps


def extract_importance(estimator, with_intercept=False):
  '''モデルから貢献度を抽出する'''
  if 'feature_importances_' in dir(estimator):
    return  estimator.feature_importances_
  elif 'coef_' in dir(estimator):
    return estimator.coef_
    if with_intercept:
      # FIXME: 戻り値のフォーマットが未決定
      return (estimator.intercept_, estimator.coef_)


def calc_metrics_var_topN(X, y, groups, estimator, x_vars_cands,
                                      max_n_feats=None, val_only=True):
  '''変数を貢献度の高いものから1つ1つ追加して、精度評価結果を出力'''
  max_n_feats = len(x_vars_cands) if max_n_feats is None else max_n_feats
  cvt = CVTrainer(X, y, groups)
  result = []
  for i in range(1, max_n_feats + 1):
    # print(i)
    x_vars_now = x_vars_cands[:i]
    cvt.x_vars = x_vars_now
    cvt.train(estimator, retrain=False)
    mms = cvt.mean_metrics
    mms['num_feats'] = i
    result.append(mms)
  result_df = pd.concat(result).reset_index()

  if val_only:
    return result_df.loc[result_df['kind'] == 'val', :]
  else:
    return result_df


def remove_x_var_not_exists(X_columns, x_vars):
  '''データセットに存在しない特徴量を除外'''
  x_vars = np.array(x_vars)
  do_exist = np.isin(x_vars, X_columns)
  if do_exist.all():
    return x_vars
  else:
    warnings.warn('Vars below are removed from x_vars since they do not exist in X columns.\n{}'.format(x_vars[~do_exist]))
    return x_vars[do_exist]


class ModelSelectRandomForestImp:
  '''RandomForestの貢献度上位の変数を使ってCV精度が最大となる上位変数を選ぶ'''

  def __init__(self, max_n_feats=100):
    self.max_n_feats = max_n_feats

  @property
  def best_metrics(self):
    return self.metrics_var_topN.iloc[self.best_n, :]

  def get_ranking(self, X, y, estimator, x_vars, groups) -> np.ndarray:

    # CVで貢献度算出
    cvt = CVTrainer(X, y, groups)
    if x_vars is not None:
      # 初期変数を絞り込む場合

      # データセットに含まれない特徴量を除外
      x_vars = remove_x_var_not_exists(X.columns, x_vars)
      cvt.x_vars = x_vars

    cvt.train(estimator, retrain=False)
    imps = extract_cvt_importance(cvt)

    # 絞り込み対象のx_varsを保存
    self.x_vars_init = cvt.x_vars

    # CVの貢献度平均値
    imps_ranking =\
      imps.groupby('x_var').mean().sort_values(by='importance',
                                                ascending=False)

    x_vars_ranking =\
      imps_ranking.index[imps_ranking['importance'] != 0].values

    return x_vars_ranking

  def select(self, X, y, groups, estimator, x_vars=None):
    '''変数の組み合わせを選択'''

    # 貢献度が0でない変数で貢献度ランキングを作成
    self.x_vars_ranking =\
      self.get_ranking(X, y, estimator, x_vars, groups)

    # n_featsの設定
    max_n_feats = min(len(self.x_vars_ranking), self.max_n_feats)
    if len(self.x_vars_ranking) > self.max_n_feats:
      max_n_feats = self.max_n_feats
      self.x_vars_ranking = self.x_vars_ranking[:max_n_feats]
    else:
      max_n_feats = len(self.x_vars_ranking)

    # 変数を貢献度の高いものから1つ1つmax_n_featsまで追加して、精度評価結果を出力
    self.metrics_var_topN = calc_metrics_var_topN(X, y, groups, estimator,
                                          self.x_vars_ranking, max_n_feats)

    # 上位の変数のみを返す
    self.best_n = self.get_best_n(self.metrics_var_topN)

    self.x_vars_best =  self.x_vars_ranking[:self.best_n + 1]
    return self.x_vars_best

  @staticmethod
  def get_best_n(vars_topN):
    # 何変数まで追加すると最大になるか
    result_val = vars_topN.loc[vars_topN['kind'] == 'val', :]
    best_n = result_val['rmse'].values.argmin()
    return best_n


class ModelSelectRandomForestSFS(ModelSelectRandomForestImp):
  '''SFSで変数リストを作成し、ひとつずつ追加したモデルで最適化

    絞り込みに相当時間がかかる:
    入力変数の数 x 絞り込む変数の数だけ処理時間がかかるので
    事前に入力行列のカラムを絞り込むか、出力変数の数を絞り込む
    目安：入力 500変数、出力：10変数
  '''
  def __init__(self, max_n_feats=10, forward=True,
                scoring='neg_mean_squared_error', cv=None):
    super(ModelSelectRandomForestSFS, self).__init__(max_n_feats)
    self.forward = forward
    self.scoring = scoring
    self.cv = cv

  def get_ranking(self, X, y, estimator, x_vars, groups=None):
      self.sfs = SFS(estimator,
                 k_features=self.max_n_feats,
                 forward=self.forward,
                 floating=False,
                 verbose=2,
                 scoring=self.scoring,
                 cv=self.cv,
                 n_jobs=-1)
      if x_vars is None:
        Xin = X
      else:
        x_vars = remove_x_var_not_exists(X.columns, x_vars)
        Xin = X[x_vars]
      self.sfs.fit(Xin, y)
      return np.array(self.sfs.k_feature_names_)
