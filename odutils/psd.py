"""Calculate FFT/PowerSpectrumDensity"""

import math
import numpy as np
import pandas as pd

from scipy import fftpack
from scipy.signal import periodogram, welch, lombscargle
from scipy.interpolate import interp1d
import warnings


def get_conv_unit(units):
    if units == 'msec':
        return 1000
    elif units == 'sec':
        return 1
    else:
        raise ValueError


def resample_evenly(x, y, fs_res, units='msec',
                    func_interp=interp1d, snap_res_unit=False):
    '''Resample y evenly using spline interpolation

    Parameters
    -------------------------
    x : array-like (N,)
        time array
    y : array-like (N,)
        value array

    '''
    f = func_interp(x, y)
    d_res = 1. / fs_res * get_conv_unit(units)
    if snap_res_unit:
        # 単位にスナップさせたい場合
        x_res_lower = math.ceil(x.min())
        x_res_upper = math.floor(x.max())
    else:
        x_res_lower = x.min()
        x_res_upper = x.max()
    x_res = np.arange(x_res_lower, x_res_upper, d_res)
    y_res = f(x_res)
    return x_res, y_res


def fft_real_amp(y, fs, use_scipy=True, f_pos_only=False):

    N = len(y)

    ind_nyq = N // 2
    if use_scipy:
        a = fftpack.fft(y) # 大きさを保つため事前に2倍にしておく
        f = fftpack.fftfreq(N, d=1/fs)
    else:
        a = np.fft.fft(y) # 大きさを保つため事前に2倍にしておく
        f = np.fft.fftfreq(N, d=1/fs)
    a[ind_nyq:] = 0
    a[:ind_nyq] *= 2
    a[0] /= 2
    if f_pos_only:
        return f[:-ind_nyq], np.abs(a[:-ind_nyq])
    else:
        return f, a


def fft(y, fs, use_scipy=True, scaling='spectrum', return_onesided=True):
    # 6/15 added

    N = len(y)

    if use_scipy:
        a = fftpack.fft(y)
        f = fftpack.fftfreq(N, d=1/fs)
    else:
        a = np.fft.fft(y)
        f = np.fft.fftfreq(N, d=1/fs)

    if scaling is None:
        pass
    elif scaling == 'amplitude':
        a = np.abs(a / N)
    elif scaling == 'spectrum':
        a = np.abs(a / N)  ** 2
    elif scaling == 'density':
        a = np.abs(a / (N * fs)) ** 2
    else:
        raise ValueError

    if return_onesided:
        ind_nyq = N // 2
        a[ind_nyq:] = 0
        a[:ind_nyq] = a[:ind_nyq] * 2
        a[0] = a[0] * 2
        return f[:ind_nyq], a[:ind_nyq]
    else:
        return f, a


def calc_psd(x, y, fs=2, method='periodogram', psd_kws=None,
                scaling='density', units='sec'):

    N = len(y)
    psd_kws = {} if psd_kws is None else psd_kws

    method_lower = method.lower()
    if method_lower == 'periodogram':
        kws = {}
        kws.update(psd_kws)
        f, pxx = periodogram(y, fs=fs, scaling=scaling, **kws)
    elif method_lower == 'welch':
        kws = {}
        kws.update(psd_kws)
        f, pxx = welch(y, fs=fs, scaling=scaling, **kws)
    elif method_lower == 'fft':
        kws = {}
        kws.update(psd_kws)
        fft = np.fft.fft(y)
        f_tmp = np.fft.fftfreq(N, d=1/fs)

        if scaling == 'amplitude':
            p_tmp = np.abs(fft) / N
        elif scaling == 'spectrum':
            p_tmp = np.abs(fft) ** 2 / N
        elif scaling == 'density':
            p_tmp = np.abs(fft) ** 2 / (N * fs)

        # 他のPSDと合わせるため、負の周波数は考えない
        max_idx = math.ceil(N / 2)
        pxx = p_tmp[:max_idx] * 2
        pxx[0] = pxx[0] / 2
        f = f_tmp[:max_idx]

    elif method_lower == 'lomb':
        # * PhysioNet HRVToolkitのlomb法が元のデータポイントの
        #   2倍の粒度で周波数変換しているので、出力粒度はそれに合わせる。
        # * 直流成分があるとエラーがでるので、最初の周波数は使わない
        # * 直流成分の分、powerの合計は小さくなる
        # * `lombscargle`にわたすのは角周波数での設定であることに注意
        warn_comm = 'With method=`lomb`, option fs={} was ignored.'
        warnings.warn(warn_comm)
        fs = 1 / np.diff(x).mean()

        nyq = fs / 2.
        f = np.linspace(0, nyq, 2 * N)[1:]
        f_ang = 2 * np.pi * f
        pxx = lombscargle(x, y, f_ang) / nyq
        pxx /= get_conv_unit(units)
        f *= get_conv_unit(units)
    else:
        raise ValueError('invalid method {}'.format(method))
    return f, pxx
