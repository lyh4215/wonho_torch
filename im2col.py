import numpy as np
from core import Module, Parameter


def im2col(x, KH, KW, stride=1, padding=0):
    N, C, H, W = x.shape

    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant"
    )

    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    cols = np.zeros((N, H_out, W_out, C, KH, KW))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + KH
            w_start = j * stride
            w_end = w_start + KW

            cols[:, i, j, :, :, :] = x_padded[:, :, h_start:h_end, w_start:w_end]

    return cols.reshape(N * H_out * W_out, C * KH * KW)


def col2im(cols, x_shape, KH, KW, stride=1, padding=0):
    N, C, H, W = x_shape

    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    cols = cols.reshape(N, H_out, W_out, C, KH, KW)

    x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + KH
            w_start = j * stride
            w_end = w_start + KW

            x_padded[:, :, h_start:h_end, w_start:w_end] += cols[:, i, j, :, :, :]

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]

    return x_padded

def im2col_fast(x, KH, KW, stride=1, padding=0):
    N, C, H, W = x.shape

    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant"
    )

    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    sN, sC, sH, sW = x_padded.strides

    shape = (N, H_out, W_out, C, KH, KW)
    strides = (sN, stride * sH, stride * sW, sC, sH, sW)

    cols = np.lib.stride_tricks.as_strided(
        x_padded,
        shape=shape,
        strides=strides
    )

    return cols.reshape(N * H_out * W_out, C * KH * KW)

def get_im2col_indices(x_shape, KH, KW, stride=1, padding=0):
    N, C, H, W = x_shape

    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    i0 = np.repeat(np.arange(KH), KW)
    i0 = np.tile(i0, C)

    i1 = stride * np.repeat(np.arange(H_out), W_out)

    j0 = np.tile(np.arange(KW), KH)
    j0 = np.tile(j0, C)

    j1 = stride * np.tile(np.arange(W_out), H_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), KH * KW).reshape(-1, 1)

    return k, i, j

def col2im_fast(cols, x_shape, KH, KW, stride=1, padding=0):
    N, C, H, W = x_shape

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    # cols: (N * H_out * W_out, C * KH * KW)
    cols_reshaped = cols.reshape(N, H_out * W_out, C * KH * KW)
    cols_reshaped = cols_reshaped.transpose(0, 2, 1)

    k, i, j = get_im2col_indices(x_shape, KH, KW, stride, padding)

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]

    return x_padded

import numpy as np
from numba import njit

from tensor import Tensor

@njit
def col2im_numba(cols, N, C, H, W, KH, KW, stride, padding):
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding))

    for n in range(N):
        for h_out in range(H_out):
            for w_out in range(W_out):
                row = n * H_out * W_out + h_out * W_out + w_out

                for c in range(C):
                    for kh in range(KH):
                        for kw in range(KW):
                            col = c * KH * KW + kh * KW + kw

                            h = h_out * stride + kh
                            w = w_out * stride + kw

                            x_padded[n, c, h, w] += cols[row, col]

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]

    return x_padded

def im2col_tensor(x, KH, KW, stride=1, padding=0):
    cols = im2col(
        x.data,
        KH,
        KW,
        stride=stride,
        padding=padding
    )

    out = Tensor(
        cols,
        _children=(x,),
        _op="im2col"
    )

    def _backward():
        N, C, H, W = x.data.shape

        dx = col2im_numba(
            out.grad,
            N, C, H, W,
            KH, KW,
            stride,
            padding
        )

        x.grad += dx

    out._backward = _backward

    return out