"""Calculate HRV features"""

import numpy as np
import pandas as pd
from .psd import resample_evenly, calc_psd
from scipy.signal import argrelmax
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
from abc import ABC, abstractmethod


f_bands = {
    'ULF': (.0, 0.0033),
    'VLF': (0.0033, 0.04),
    'LF': (0.04, 0.15),
    'HF': (0.15, 0.4),
    }


def plot_powerband(ax, f, pxx, f_bands=f_bands, units='sec', hrvs=None):
    ax.plot(f, pxx, color='grey', lw=.7)
    ax.set(**{'title': 'PSD',
                'xlabel': 'Freqency [Hz]',
                'ylabel': '{} ** 2 / Hz'.format(units)})
    if f_bands is not None:
        for i, (b_nm, b_range) in enumerate(f_bands.items()):
            b_lower, b_upper = b_range
            isinband = np.logical_and(f >= b_lower, f <= b_upper)
            ax.fill_between(f[isinband], pxx[isinband],
                            color='C{}'.format(i), label=b_nm, alpha=.7)
        if hrvs is not None:
            info = '\n'.join(['{}={:.5f}'.format(k, v)
                for k, v in hrvs.items()])
            ax.text(f.max() * 0.05, pxx.max() * 0.3, info)
        ax.legend()
    return ax


def get_conv_unit(units):
    if units == 'msec':
        return 1000
    elif units == 'sec':
        return 1
    else:
        raise ValueError


def integ_sum(pxx, df):
    return np.sum(pxx)


def freq_band_powers(f, pxx, integ_func=np.trapz, f_bands=f_bands,
                        key_prefix=''):
    pwrs = {}
    for k, band in f_bands.items():
        lower, upper = band
        x = np.logical_and(f >= lower, f <= upper)
        pwrs[key_prefix + k] = integ_func(pxx[x], f[x])
    return pwrs


def calc_peakfreq(f, pxx, top_n=5):
    ind = argrelmax(pxx)[0]
    pxx_peaks = pxx[ind]
    f_peaks = f[ind]
    top_n = min(top_n, len(pxx_peaks))
    top_ind = np.flip(pxx_peaks.argsort())[:top_n]
    return f_peaks[top_ind], pxx_peaks[top_ind]


def calc_peakfreq_in_band(f, pxx, f_bands=f_bands):
    peak_freq = {}
    for k, band in f_bands.items():
        lower, upper = band
        isin_band = np.logical_and(f >= lower, f <= upper)
        pxx_temp = pxx.copy()
        pxx_temp[np.logical_not(isin_band)] = -np.inf
        peak_freq[k + '_peak'] = f[np.argmax(pxx_temp)]
    return peak_freq


def calc_pwr(f, pxx, integ_func=np.trapz, f_bands=f_bands):
    pwrs = freq_band_powers(f, pxx, integ_func=np.trapz, f_bands=f_bands)
    pwrs['LF/HF'] = pwrs['LF'] / pwrs['HF']
    pwrs['LF_ratio'] = pwrs['LF'] / (pwrs['LF'] + pwrs['HF'])
    pwrs['HF_ratio'] = pwrs['HF'] / (pwrs['LF'] + pwrs['HF'])
    return pwrs



class HRVTime:

    TIME_UNITS_CONV = {'sec': 1.0, 'msec': 1000.0}

    def __init__(self, thresh_d=None, units='sec',
                        consider_seq_missing=True):

        self.conv_unit = HRVTime.TIME_UNITS_CONV[units]  # 単位

        self.consider_seq_missing = consider_seq_missing

         # 一致しているとみなす誤差の最小[sec]
        if thresh_d is None:
            self.thresh_d = 0.005 * self.conv_unit
        else:
            self.thresh_d = thresh_d

    def _eval_valid_ss(self, ts, rri):
        """有効な連続か評価する"""
        ts_diff = np.diff(ts)
        thresh = self.thresh_d * self.conv_unit
        isvalid_ss = np.abs(rri[1:] - ts_diff) < thresh
        # print(np.abs(rri[1:] - ts_diff))
        # 先頭RRIは隣接がないので、有効でないとする
        is_valid_ss = np.insert(isvalid_ss, 0, False)
        return is_valid_ss

    def calc(self, ts, rri):
        measures = {}

        m_stats = self._calc_stats_measures(ts, rri)
        measures.update(m_stats)

        m_ss = self._calc_ss_measures(ts, rri, self.consider_seq_missing)
        measures.update(m_ss)

        return measures

    def _calc_stats_measures(self, ts, rri):
        rri_mean = np.mean(rri) # Mean RR
        rri_sd = np.std(rri)  # SDNN
        rri_var = np.var(rri)  # var
        rri_cv = np.std(rri) / np.mean(rri) * 100.0  # CVRR
        minutes = 60 * self.conv_unit
        hr_mean = minutes / rri_mean  # Mean HR
        return dict(
                    rri_mean=rri_mean,
                    rri_sd=rri_sd,
                    rri_var=rri_var,
                    rri_cv=rri_cv,
                    hr_mean=hr_mean,
                    )

    def _calc_ss_measures(self, ts, rri, consider_seq_missing):

        if consider_seq_missing:
            # 欠損などにより連続していないRRIがあるので、連続するRRIか評価
            is_valid_ss = self._eval_valid_ss(ts, rri)
            # 連続するRRIのみで差分をとる
            valid_ind = np.where(is_valid_ss)[0]
            rri_valid_ss = rri[valid_ind]
            rri_valid_ss_prev = rri[valid_ind - 1]
        else:
            rri_valid_ss = rri[1:]
            rri_valid_ss_prev = rri[:-1]

        # import ipdb;ipdb.set_trace()

        self.rri_valid_ss = rri_valid_ss
        self.rri_valid_ss_prev = rri_valid_ss_prev

        d_ss = rri_valid_ss - rri_valid_ss_prev

        # 統計量を算出
        n_valid_ss = rri_valid_ss.size
        rri_length = len(rri)
        ratio_valid_ss = n_valid_ss / rri_length

        rmssd = np.sqrt(np.mean(d_ss ** 2))  # RMSSD

        thresh = 0.05 * self.conv_unit
        nn50_count = (d_ss > thresh).sum()  # NN50 count
        pnn50 = nn50_count / n_valid_ss  # pNN50

        return dict(
                    rri_length=rri_length,
                    n_valid_ss=n_valid_ss,
                    ratio_valid_ss=ratio_valid_ss,
                    rmssd=rmssd,
                    nn50_count=nn50_count,
                    pnn50=pnn50
                    )


class HRVFreq:

    def __init__(self, fs, method='periodogram', do_resample=False,
                    fs_res=None, psd_kws=None, units='sec'):

        '''
        Parameters
        ------------
        do_resample bool, default=False

        fs_res int, default=None
          resampling frequency. Use if do_resample=True.


        '''
        self.fs = fs
        self.method = method
        if do_resample:
          if fs_res is None:
            msg = 'If do_resample is True you should specify fs_res.'
            raise ValueError(msg)

        self.do_resample = do_resample
        self.fs_res = fs_res

        self.psd_kwd = psd_kws
        self.units = units

        self.pxx = None
        self.f = None

    def calc(self, x, y):

        hrvs = {}

        if self.do_resample:
            x_r, y_r = resample_evenly(x, y, self.fs_res, units=self.units)
        else:
            x_r, y_r = x, y

        self.x_r, self.y_r = x_r, y_r

        fs = self.fs_res if self.do_resample else self.fs
        f, pxx = calc_psd(x_r, y_r, fs=fs,
                            method=self.method, psd_kws=self.psd_kwd,
                            units=self.units)
        self.f, self.pxx = f, pxx

        pwrs = calc_pwr(f, pxx)
        peaks = calc_peakfreq_in_band(f, pxx)
        peak_f_top, peak_pxx_top = calc_peakfreq(f, pxx)

        top_peaks = {}
        for i, peak_f in enumerate(peak_f_top):
            top_peaks['top{}_peak_f'.format(i + 1)] = peak_f

        hrvs.update(pwrs)
        hrvs.update(peaks)
        hrvs.update(top_peaks)

        return hrvs
