"""TomoLarge — host-chunked USFFT Radon for volumes too big for GPU-only Tomo.

Vendored from radon_large/tomo_large.py.  Stages small pieces of the padded
(2N × 2N) frequency-domain buffer through the GPU while keeping the big
`fde` and `sino` arrays on the HOST.  Peak GPU memory is proportional to
the chunk sizes rather than to (2N)².
"""
from __future__ import annotations

import numpy as np
import cupy as cp

from processing.kernels import gather_kernel1


class TomoLarge:
    """Radon transform via USFFT with host-staged chunking.

    R(obj, chunks) is the only entry-point used by model_radon_large.py;
    the adjoint RT is included for round-trip debugging.
    """

    def __init__(self, n, theta, rotation_axis=None):
        """USFFT parameter setup (host-side); no per-call GPU allocations."""
        eps = 1e-3
        mu  = -np.log(eps) / (2 * n * n)
        m   = int(np.ceil(2 * n * 1 / np.pi *
                          np.sqrt(-mu * np.log(eps) + (mu * n) ** 2 / 4)))

        ntheta = len(theta)

        # phi is the Gaussian pre-multiplication kernel evaluated on the
        # (n × n) grid; kept host-side because obj is host-side.
        t = np.linspace(-1 / 2, 1 / 2, n, endpoint=False).astype("float32")
        dx, dy = np.meshgrid(t, t)
        phi = np.exp(mu * (n * n) * (dx * dx + dy * dy)).astype("complex64") * (1 - n % 4)

        c1dfftshift = (1 - 2 * ((cp.arange(1, n + 1) % 2))).astype("int8")
        c2dtmp      = 1 - 2 * ((np.arange(1, 2 * n + 1) % 2)).astype("int8")
        c2dfftshift = np.outer(c2dtmp, c2dtmp)

        # Sample-point coordinates on the doubled Fourier grid, indexed by
        # (theta_id, r).  Sorted into 2-D chunks so each gather kernel launch
        # touches a small region of fde.
        x = np.empty([ntheta * n], dtype="float32")
        y = np.empty([ntheta * n], dtype="float32")
        theta32 = theta.astype("float32")
        for k in range(ntheta):
            r  = np.arange(-n / 2, n / 2, dtype="float32") / n
            x0 =  np.cos(theta32[k]) * r
            y0 = -np.sin(theta32[k]) * r
            x[k * n:(k + 1) * n] = x0
            y[k * n:(k + 1) * n] = y0

        # Clamp into [-0.5, 0.5) so floor(2n·x)+n lands in [0, 2n).
        x = np.clip(x, -0.5,           0.5 - 1e-5)
        y = np.clip(y, -0.5,           0.5 - 1e-5)

        self.x      = x
        self.y      = y
        self.n      = n
        self.ntheta = ntheta
        self.theta  = theta32
        self.mua    = cp.array([mu], dtype="float32")
        self.m      = m
        self.phi    = phi
        self.c1dfftshift = c1dfftshift
        self.c2dfftshift = c2dfftshift
        # rotation_axis is accepted for API compatibility but unused here
        # (rotation is centred at N/2 via the sample-point formulas).
        self.rotation_axis = rotation_axis

    def _sort_into_chunks(self, chunk_xy):
        """Precompute per-chunk sample lists for a given XY chunk size."""
        n = self.n
        f_indx = np.floor(2 * n * self.x).astype("int64") + n
        f_indy = np.floor(2 * n * self.y).astype("int64") + n
        qid = (f_indy // chunk_xy) * (2 * n // chunk_xy) + f_indx // chunk_xy

        idx = np.argsort(qid)
        x_s, y_s, qid_s = self.x[idx], self.y[idx], qid[idx]

        nel = np.zeros((2 * n // chunk_xy) ** 2, dtype="int64")
        change_points = np.flatnonzero(np.diff(qid_s, prepend=qid_s[0] - 1))
        run_lengths = np.diff(np.append(change_points, len(qid_s)))
        nel[qid_s[change_points]] = run_lengths
        return x_s, y_s, nel, idx

    def _get_st_end(self, indx, indy, chunk_xy):
        n, m = self.n, self.m
        stx = int(max(0, indx * chunk_xy - m))
        endx = int(min((indx + 1) * chunk_xy + m + 1, 2 * n))
        sty = int(max(0, indy * chunk_xy - m))
        endy = int(min((indy + 1) * chunk_xy + m + 1, 2 * n))
        return [stx, endx, sty, endy]

    # ---------- forward Radon ------------------------------------------------
    def R(self, obj, chunks):
        """(nz, n, n) obj → (ntheta, nz, n) sinogram; obj/sino live on host.

        chunks = [CHUNK_N, CHUNK_THETA, CHUNK_XY] — chunk sizes for the
        1-D FFTs, angle grouping, and gather bin size respectively.
        """
        chunk_n, chunk_theta, chunk_xy = chunks
        m, mua, phi, c1dfftshift, c2dfftshift = (
            self.m, self.mua, self.phi, self.c1dfftshift, self.c2dfftshift)
        n, ntheta = self.n, self.ntheta
        nz = obj.shape[0]

        x_s, y_s, nel, idx = self._sort_into_chunks(chunk_xy)

        # --- FFT along x, then along y, both host-staged in strips ---------
        fde = np.empty([nz, 2 * n, 2 * n], dtype="complex64")
        fde[:] = 0

        for k in range(n // chunk_n):
            st, end = k * chunk_n, (k + 1) * chunk_n
            obj0 = cp.array(obj[:, st:end])
            phi0 = cp.array(phi[st:end])
            c2d0 = cp.array(c2dfftshift[st:end])
            fde0 = phi0[None] * obj0
            fde0 = cp.pad(fde0, ((0, 0), (0, 0), (n // 2, n // 2)))
            fde0 = cp.fft.fft(fde0 * c2d0[None], axis=-1) * c2d0[None]
            fde[:, n // 2 + st : n // 2 + end] = fde0.get()

        for k in range(2 * n // chunk_n):
            st, end = k * chunk_n, (k + 1) * chunk_n
            fde0 = cp.array(fde[:, :, st:end])
            c2d0 = cp.array(c2dfftshift[:, st:end])
            fde0 = cp.fft.fft(fde0 * c2d0[None], axis=1) * c2d0[None]
            fde[:, :, st:end] = fde0.get()

        # --- NUFFT gather per chunk -----------------------------------------
        sino = np.empty([ntheta * nz * n], dtype="complex64")
        offset = 0
        n_chunk_xy = 2 * n // chunk_xy
        for indy in range(n_chunk_xy):
            for indx in range(n_chunk_xy):
                ind = indy * n_chunk_xy + indx
                if nel[ind] == 0:
                    continue
                stx, endx, sty, endy = self._get_st_end(indx, indy, chunk_xy)
                fde_d = cp.ascontiguousarray(cp.array(fde[:, sty:endy, stx:endx]))
                for zc in range(0, nz):
                    x0 = cp.array(x_s[offset : offset + nel[ind]])
                    y0 = cp.array(y_s[offset : offset + nel[ind]])
                    sino0 = cp.zeros([nel[ind]], dtype="complex64")
                    gather_kernel1(
                        (int(cp.ceil(nel[ind] / 1024)),),
                        (1024,),
                        (sino0, fde_d[zc], x0, y0, m, mua, nel[ind],
                         stx, endx, sty, endy, n, 0),
                    )
                    sino_full_idx = idx[offset : offset + nel[ind]]
                    # Interleave sample index with z: dest = t*nz*n + z*n + r
                    # but our sample index encodes (theta, r) as t*n + r.
                    t_of  = sino_full_idx // n
                    r_of  = sino_full_idx %  n
                    flat  = t_of * nz * n + zc * n + r_of
                    sino[flat] = sino0.get()
                offset += nel[ind]

        sino = sino.reshape([ntheta, nz, n])

        # --- 1-D IFFT along the sample axis and normalisation ---------------
        for k in range(ntheta // chunk_theta):
            st, end = k * chunk_theta, (k + 1) * chunk_theta
            sino0 = cp.array(sino[st:end])
            sino0 = cp.fft.ifft(c1dfftshift * sino0) * c1dfftshift
            sino0 /= 4 * n * np.sqrt(n * ntheta)
            sino[st:end] = sino0.get()

        return sino
