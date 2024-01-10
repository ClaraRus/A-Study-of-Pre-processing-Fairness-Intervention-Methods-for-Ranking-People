import numpy as np
import scipy.optimize as optim

import src.modules.LFR.helpers as lfr_helpers


class LFR():
    """Learning fair representations is a pre-processing technique that finds a
    latent representation which encodes the data well but obfuscates information
    about protected attributes [2]_.
    References:
        .. [2] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,  "Learning
           Fair Representations." International Conference on Machine Learning,
           2013.
    Based on code from https://github.com/zjelveh/learning-fair-representations
    """

    def __init__(self,
                 k=5,
                 Ax=0.01,
                 Ay=1.0,
                 Az=50.0,
                 print_interval=250,
                 verbose=0,
                 logs_path='',
                 seed=None):
        """
        Args:
            k (int, optional): Number of prototypes.
            Ax (float, optional): Input recontruction quality term weight.
            Az (float, optional): Fairness constraint term weight.
            Ay (float, optional): Output prediction error.
            print_interval (int, optional): Print optimization objective value
                every print_interval iterations.
            verbose (int, optional): If zero, then no output.
            seed (int, optional): Seed to make `predict` repeatable.
        """

        self.seed = seed
        self.logs_path = logs_path

        self.k = k
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az

        self.print_interval = print_interval
        self.verbose = verbose

        self.w = None
        self.prototypes = None
        self.learned_model = None

    def fit(self, dataset, features_cols, label_col, sensitive_col, maxiter=5000, maxfun=5000):
        """Compute the transformation parameters that leads to fair representations.
        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.
            features_cols: columns names of the dataset to be transformed.
            label_col: column name of the label.
            sensitive_col: column name of the sensitive data.
            maxiter (int): Maximum number of iterations.
            maxfun (int): Maxinum number of function evaluations.
        Returns:
            LFR: Returns self.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        features_dim = len(features_cols)
        groups = dataset[sensitive_col].unique()
        group_1 = dataset[dataset[sensitive_col] == groups[0]]
        group_2 = dataset[dataset[sensitive_col] == groups[1]]
        features_group_1 = group_1[features_cols].values
        features_group_2 = group_2 [features_cols].values

        labels_group_1 = group_1[label_col].values
        labels_group_2 = group_2[label_col].values

        # Initialize the LFR optim objective parameters
        parameters_initialization = np.random.uniform(size=self.k + features_dim * self.k)
        bnd = [(0, 1)] * self.k + [(None, None)] * features_dim * self.k
        lfr_helpers.LFR_optim_objective.steps = 0

        self.learned_model = \
        optim.fmin_l_bfgs_b(lfr_helpers.LFR_optim_objective, x0=parameters_initialization, epsilon=1e-5,
                            args=(features_group_1, features_group_2,
                                  labels_group_1, labels_group_2, features_dim, self.k, self.Ax,
                                  self.Ay, self.Az, self.print_interval, self.logs_path, self.verbose),
                            bounds=bnd, approx_grad=True, maxfun=maxfun,
                            maxiter=maxiter, disp=self.verbose)[0]
        self.w = self.learned_model[:self.k]
        self.prototypes = self.learned_model[self.k:].reshape((self.k, features_dim))

        return self

    def transform(self, dataset, features_cols):
        """Transform the dataset using learned model parameters.
        Args:
            dataset (dataframe): Dataset containing features to be transformed.
            features_cols: columns names of the dataset to be transformed
        Returns:
            dataset (dataframe): Transformed Dataset.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        features = dataset[features_cols].values
        _, features_hat, labels_hat = lfr_helpers.get_xhat_y_hat(self.prototypes, self.w, features)

        return features_hat