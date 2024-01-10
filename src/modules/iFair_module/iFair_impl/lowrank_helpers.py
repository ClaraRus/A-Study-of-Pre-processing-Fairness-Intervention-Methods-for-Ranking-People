"""
Implementation of the ICDE 2019 paper
iFair_module: Learning Individually Fair Data Representations for Algorithmic Decision Making
__url__: https://ieeexplore.ieee.org/abstract/document/8731591
__author__: Preethi Lahoti
__email__: plahoti@mpi-inf.mpg.de
"""
from __future__ import division
from numba.decorators import jit
import numpy as np
from numpy import linalg
from sklearn import preprocessing
import sklearn.metrics.pairwise as pairwise

#d(x_n,v_k,alpha) = sum_limits_{m = 1}^{M} alpha_m (x_nm - v_km)^2
@jit
def dist_prototype(X, v, alpha, N, M, k):
    dists = np.zeros((N, k))
    for i in range(N):
        for m in range(M):
            for j in range(k):
                dists[i, j] += (X[i, m] - v[j, m]) * (X[i, m] - v[j, m]) * alpha[m]
    return dists

#d(x_i,x_j,alpha) = sum_limits_{m = 1}^{M} alpha_m (x_im - x_jm)^2
#todo: can be optimized
@jit
def dist_pairwise(X, alpha, N, M):
    dists = np.zeros((N, N))
    for i in range(N):
        for m in range(M):
            for j in range(N):
                dists[i, j] += (X[i, m] - X[j, m]) * (X[i, m] - X[j, m]) * alpha[m]
    return dists

#U_{n,k} = Pr(Z = k | x_n) = exp(-d(x_n,v_k))/\sum_limits_{j=1}^K exp(-d(x_n,v_j))
@jit
def compute_U_nk(dists, N, k):
    U_nk = np.zeros((N, k))
    exp = np.zeros((N, k))
    denom = np.zeros(N)
    for i in range(N):
        for j in range(k):
            exp[i, j] = np.exp(-1 * dists[i, j])
            denom[i] += exp[i, j]
        for j in range(k):
            if denom[i]:
                U_nk[i, j] = exp[i, j] / denom[i]
            else:
                U_nk[i, j] = exp[i, j] / 1e-6
    # U_nk is a probabability vector with values in [0,1]
    # - clipping to avoid numerical errors
    U_nk = np.clip(U_nk, a_min=0, a_max=1)
    return U_nk


def replaceNaNwithMean(X):
    """
    Replaces NaN with column mean
    :param X: 2D array
    :return: 2D array
    """
    # Obtain mean of columns as you need, nanmean is just convenient.
    col_mean = np.nanmean(X, axis=0)
    # - guarding againts a column where all values are nan
    col_mean[np.isnan(col_mean)] = 0

    # Find indicies that you need to replace
    inds = np.where(np.isnan(X))

    # Place column means in the indices. Align the arrays using take
    X[inds] = np.take(col_mean, inds[1])

    return X


#X_hat = U_nk * V_kn
#L_x = sum_limits_{n = 1}^{N} (x_n - x_n_hat)^2
@jit
def compute_X_hat(U_nk, v, N, M, k):
    X_hat = np.zeros((N, M))
    for i in range(N):
        for m in range(M):
            for j in range(k):
                X_hat[i, m] += U_nk[i, j] * v[j, m]

    # check if X_hat has NaN or Infinte
    if np.isfinite(X_hat).all() == False:
        # - steps to avoid numerical errors
        # -- clips values (including infinite) to specified range
        X_hat = np.clip(X_hat, a_min=1e-5, a_max=1e+5)
        X_hat = replaceNaNwithMean(X_hat)

    return X_hat



@jit
def iFair(params, X, batch_indx, D_X_f=0, k=10, A_x=1e-4, A_z=1e-4, results=0, recompute_D_X_f = False):
    iFair.iters += 1
    N, M = X.shape
    alpha = params[:M]
    v = np.matrix(params[(2 * M) + k:]).reshape((k, M))
    distances_prototypes = dist_prototype(X, v, alpha, N, M, k)
    U_nk = compute_U_nk(distances_prototypes, N, k)
    X_hat = compute_X_hat(U_nk, v, N, M, k)
    if results:
        return X_hat
    else:
        # computing reconstruction loss L_x
        x_hat_batch = X_hat[batch_indx[0]:batch_indx[1]]
        x_batch = X[batch_indx[0]:batch_indx[1]]
        dxf_batch = D_X_f[batch_indx[0]:batch_indx[1]]

        L_x = linalg.norm(x_batch - x_hat_batch)
        # computing individual fairness loss L_z
        D_X_f_hat = pairwise.euclidean_distances(x_hat_batch,X_hat)
        if recompute_D_X_f:
            D_X_f = dist_pairwise(X, alpha, N, M)
        L_z = linalg.norm(dxf_batch - D_X_f_hat)
        criterion = A_x * L_x + A_z * L_z
        if iFair.iters % 100 == 0:
            print(iFair.iters, L_x, L_z, criterion)
        return criterion

@jit
def predict(params, X, k):
    N, M = X.shape
    alpha = params[:M]
    v = np.matrix(params[(2 * M) + k:]).reshape((k, M))
    distances_prototypes = dist_prototype(X, v, alpha, N, M, k)
    U_nk = compute_U_nk(distances_prototypes, N, k)
    X_hat = compute_X_hat(U_nk, v, N, M, k)
    return X_hat

iFair.iters = 1