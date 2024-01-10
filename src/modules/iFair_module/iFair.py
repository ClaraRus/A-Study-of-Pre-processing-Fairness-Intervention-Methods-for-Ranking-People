"""
Implementation of the ICDE 2019 paper
iFair_module: Learning Individually Fair Data Representations for Algorithmic Decision Making
url: https://ieeexplore.ieee.org/document/8731591
citation:
@inproceedings{DBLP:conf/icde/LahotiGW19,
  author    = {Preethi Lahoti and
               Krishna P. Gummadi and
               Gerhard Weikum},
  title     = {iFair_module: Learning Individually Fair Data Representations for Algorithmic
               Decision Making},
  booktitle = {35th {IEEE} International Conference on Data Engineering, {ICDE} 2019,
               Macao, China, April 8-11, 2019},
  pages     = {1334--1345},
  publisher = {{IEEE}},
  year      = {2019},
  url       = {https://doi.org/10.1109/ICDE.2019.00121},
  doi       = {10.1109/ICDE.2019.00121},
  timestamp = {Wed, 16 Oct 2019 14:14:56 +0200},
  biburl    = {https://dblp.org/rec/conf/icde/LahotiGW19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


__author__: Preethi Lahoti
__email__: plahoti@mpi-inf.mpg.de
"""
import os
import pickle
import numpy as np
from scipy.optimize import minimize

from src.modules.iFair_module.iFair_impl.lowrank_helpers import iFair as ifair
from src.modules.iFair_module.iFair_impl.lowrank_helpers import predict as ifair_predict
from src.utils.utils import compute_euclidean_distances


class iFair:

    def __init__(self, out_path, k=2, A_x=1e-2, A_z=1.0, max_iter=1000, nb_restarts=3):
        self.k = k
        self.A_x = A_x
        self.A_z = A_z
        self.max_iter = max_iter
        self.nb_restarts = nb_restarts
        self.opt_params = None
        self.out_path = out_path

    def fit(self, X_train, i, qid, batch_size, nonsensitive_column_indices=None):
        """
        Learn the model using the training data. iFair_module.py._func
        :param X:     Training data. Expects last column of the matrix X to be the protected attribute.
        """

        ifair_func = ifair
        print('Fitting iFair_module...')
        if not nonsensitive_column_indices:
            nonsensitive_column_indices = list(range(0, X_train.shape[1] - 1))

        if qid is not None:
            euclidean_dist_dir = os.path.join(self.out_path, str(i), 'train', str(qid))
        else:
            euclidean_dist_dir = os.path.join(self.out_path,  str(i), 'train')
        print(euclidean_dist_dir)
        if not os.path.exists(euclidean_dist_dir):
            print("Compute Distances")
            D_X_F = compute_euclidean_distances(X_train, euclidean_dist_dir, None,
                                                nonsensitive_column_indices)
        else:
            with open(os.path.join(euclidean_dist_dir, 'euclidean_distance.pkl'), 'rb') as f:
                D_X_F = pickle.load(f)
        l = len(nonsensitive_column_indices)

        P = X_train.shape[1]
        min_obj = None
        opt_params = None
        for i in range(self.nb_restarts):
            x0_init = np.random.uniform(size=P * 2 + self.k + P * self.k)
            # setting protected column weights to epsilon
            ## assumes that the column indices from l through P are protected and appear at the end
            for i in range(l, P, 1):
                x0_init[i] = 0.0001
            bnd = [(None, None) if (i < P * 2) or (i >= P * 2 + self.k) else (0, 1)
                   for i in range(len(x0_init))]
            if X_train.shape[0] <= batch_size or batch_size == 0:
                batches = 1
                batch_size = X_train.shape[0]
            else:
                batches = int(X_train.shape[0] / batch_size) + 1
            for i in range(batches):
                if batches == 1 and i == 1:
                    break

                print("Start batch " + str(i))
                start_index = i * batch_size
                end_index = (i + 1) * batch_size
                if end_index > X_train.shape[0]:
                    end_index = X_train.shape[0]

                opt_result = minimize(ifair_func, x0_init,
                                      args=(X_train, (start_index, end_index), D_X_F, self.k, self.A_x, self.A_z, 0),
                                      method='L-BFGS-B',
                                      jac=False,
                                      bounds=bnd,
                                      options={'maxiter': self.max_iter,
                                               'maxfun': self.max_iter,
                                               'eps': 1e-3})

                x0_init = opt_result.x

            if (min_obj is None) or (opt_result.fun < min_obj):
                min_obj = opt_result.fun
                opt_params = opt_result.x

        self.opt_params = opt_params

    def transform(self, X):
        X_hat = ifair_predict(self.opt_params, X, k=self.k)
        return X_hat