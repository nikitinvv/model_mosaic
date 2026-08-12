"""Wavelet-FFT (FW) ring removal for GPU sinograms.

Vendored verbatim from tomocupy/src/tomocupy/processing/remove_stripe.py
(functions _reflect / _mypad / afb1d / sfb1d, classes DWTForward /
DWTInverse, and remove_stripe_fw).  Kept local because tomocupy's
__init__ pulls in numexpr + full config, which we don't want to depend
on from a leaf preprocessing step — every other mosaic_modeling.processing
module follows the same self-contained pattern.
"""
# *************************************************************************** #
#                  Copyright (C) 2022, UChicago Argonne, LLC                  #
#                           All Rights Reserved                               #
#                         Software Name: Tomocupy                             #
#                     By: Argonne National Laboratory                         #
#                           OPEN SOURCE LICENSE                               #
# *************************************************************************** #

import cupy as cp
import pywt


__all__ = ['DWTForward', 'DWTInverse', 'afb1d', 'sfb1d', 'remove_stripe_fw']


def _reflect(x, minx, maxx):
    x = cp.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = cp.fmod(x - minx, rng_by_2)
    normed_mod = cp.where(mod < 0, mod + rng_by_2, mod)
    out = cp.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return cp.array(out, dtype=x.dtype)


def _mypad(x, pad, value=0):
    if pad[0] == 0 and pad[1] == 0:
        m1, m2 = pad[2], pad[3]
        l = x.shape[-2]
        xe = _reflect(cp.arange(-m1, l + m2, dtype='int32'), -0.5, l - 0.5)
        return x[:, :, xe]
    elif pad[2] == 0 and pad[3] == 0:
        m1, m2 = pad[0], pad[1]
        l = x.shape[-1]
        xe = _reflect(cp.arange(-m1, l + m2, dtype='int32'), -0.5, l - 0.5)
        return x[:, :, :, xe]


def afb1d(x, h0, h1='zero', dim=-1):
    """1D analysis filter bank: stride-2 convolution along dim, all
    channels in parallel.  Output channels interleaved
    [lo_0, hi_0, lo_1, hi_1, ...]."""
    C = x.shape[1]
    d = dim % 4
    N = x.shape[d]
    h0f = h0.flatten()
    h1f = h1.flatten()
    L = h0f.size
    outsize = pywt.dwt_coeff_len(N, L, mode='symmetric')
    p = 2 * (outsize - 1) - N + L
    pad = (0, 0, p // 2, (p + 1) // 2) if d == 2 else (p // 2, (p + 1) // 2, 0, 0)
    x = _mypad(x, pad=pad)
    B = x.shape[0]
    if d == 3:
        H = x.shape[2]
        out = cp.empty((B, C, 2, H, outsize), dtype='float32')
        sl0 = x[:, :, :, 0:2 * outsize:2]
        out[:, :, 0] = h0f[0] * sl0
        out[:, :, 1] = h1f[0] * sl0
        for j in range(1, L):
            sl = x[:, :, :, j:j + 2 * outsize:2]
            out[:, :, 0] += h0f[j] * sl
            out[:, :, 1] += h1f[j] * sl
    else:
        W = x.shape[3]
        out = cp.empty((B, C, 2, outsize, W), dtype='float32')
        sl0 = x[:, :, 0:2 * outsize:2, :]
        out[:, :, 0] = h0f[0] * sl0
        out[:, :, 1] = h1f[0] * sl0
        for i in range(1, L):
            sl = x[:, :, i:i + 2 * outsize:2, :]
            out[:, :, 0] += h0f[i] * sl
            out[:, :, 1] += h1f[i] * sl
    return out.reshape(B, 2 * C, *out.shape[3:])


def sfb1d(lo, hi, g0, g1='zero', dim=-1):
    """1D synthesis filter bank: scatter-add (upsampled transposed conv)."""
    C = lo.shape[1]
    d = dim % 4
    g0f = g0.flatten()
    g1f = g1.flatten()
    L = g0f.size
    B = lo.shape[0]
    if d == 3:
        H, W = lo.shape[2], lo.shape[3]
        wi = (W - 1) * 2 + L
        out = cp.zeros((B, C, H, wi), dtype='float32')
        for j in range(L):
            out[:, :, :, j:j + 2 * W:2] += g0f[j] * lo + g1f[j] * hi
        return out[:, :, :, (L - 2):wi - (L - 2)]
    else:
        H, W = lo.shape[2], lo.shape[3]
        hi_size = (H - 1) * 2 + L
        out = cp.zeros((B, C, hi_size, W), dtype='float32')
        for i in range(L):
            out[:, :, i:i + 2 * H:2, :] += g0f[i] * lo + g1f[i] * hi
        return out[:, :, (L - 2):hi_size - (L - 2), :]


class DWTForward:
    def __init__(self, wave='db1'):
        wave = pywt.Wavelet(wave)
        h0_col, h1_col = wave.dec_lo, wave.dec_hi
        h0_row, h1_row = h0_col, h1_col
        self.h0_col = cp.array(h0_col).astype('float32')[::-1].reshape((1, 1, -1, 1))
        self.h1_col = cp.array(h1_col).astype('float32')[::-1].reshape((1, 1, -1, 1))
        self.h0_row = cp.array(h0_row).astype('float32')[::-1].reshape((1, 1, 1, -1))
        self.h1_row = cp.array(h1_row).astype('float32')[::-1].reshape((1, 1, 1, -1))

    def apply(self, x):
        lohi = afb1d(x, self.h0_row, self.h1_row, dim=3)
        y = afb1d(lohi, self.h0_col, self.h1_col, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        x = cp.ascontiguousarray(y[:, :, 0])
        yh = cp.ascontiguousarray(y[:, :, 1:])
        return x, yh


class DWTInverse:
    def __init__(self, wave='db1'):
        wave = pywt.Wavelet(wave)
        g0_col, g1_col = wave.rec_lo, wave.rec_hi
        g0_row, g1_row = g0_col, g1_col
        self.g0_col = cp.array(g0_col).astype('float32').reshape((1, 1, -1, 1))
        self.g1_col = cp.array(g1_col).astype('float32').reshape((1, 1, -1, 1))
        self.g0_row = cp.array(g0_row).astype('float32').reshape((1, 1, 1, -1))
        self.g1_row = cp.array(g1_row).astype('float32').reshape((1, 1, 1, -1))

    def apply(self, coeffs):
        yl, yh = coeffs
        lo_hi = sfb1d(cp.concatenate([yl,        yh[:, :, 1]], axis=1),
                      cp.concatenate([yh[:, :, 0], yh[:, :, 2]], axis=1),
                      self.g0_col, self.g1_col, dim=2)
        yl = sfb1d(lo_hi[:, :1], lo_hi[:, 1:], self.g0_row, self.g1_row, dim=3)
        return yl


def remove_stripe_fw(data, sigma, wname, level):
    """FW (wavelet-FFT) ring removal.

    Input `data` shape: (nproj, nz, ni), cupy float32.  Returns the
    ring-suppressed array in the same shape/dtype.
    """
    [nproj, nz, ni] = data.shape
    nproj_pad = nproj + nproj // 8

    xfm = DWTForward(wave=wname)
    ifm = DWTInverse(wave=wname)

    cc = []
    sli = cp.zeros([nz, 1, nproj_pad, ni], dtype='float32')
    sli[:, 0, (nproj_pad - nproj) // 2:(nproj_pad + nproj) // 2] = \
        data.astype('float32').swapaxes(0, 1)
    for k in range(level):
        sli, c = xfm.apply(sli)
        cc.append(c)
        band = cc[k][:, 0, 1]
        _, my, mx = band.shape
        fcV = cp.fft.rfft(band, axis=1)
        myr = my // 2 + 1
        y_hat = cp.fft.ifftshift((cp.arange(-my, my, 2) + 1) / 2)[:myr]
        damp = -cp.expm1(-y_hat ** 2 / (2 * sigma ** 2))
        fcV *= damp[:, None]
        cc[k][:, 0, 1] = cp.fft.irfft(fcV, my, axis=1)

    for k in range(level)[::-1]:
        shape0 = cc[k][0, 0, 1].shape
        sli = sli[:, :, :shape0[0], :shape0[1]]
        sli = ifm.apply((sli, cc[k]))

    out = sli[:, 0, (nproj_pad - nproj) // 2:(nproj_pad + nproj) // 2, :ni] \
        .astype(data.dtype)
    return out.swapaxes(0, 1)
