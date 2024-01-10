import os.path
import pandas as pd
from src.utils.utils import create_train_split


class Dataset:
    def __init__(self, path, dataset_name, query_col, sensitive_col, score_col, features_col, DV, MED, k_fold, ratio_split,
                 version=None, th=0):
        self.path = path
        self.version = version
        self.dataset_name = dataset_name

        self.query_col = query_col
        self.score_col = score_col
        self.features_cols = features_col
        self.sensitive_col = sensitive_col
        self.k_fold = k_fold
        self.ratio_split = ratio_split

        self.MED = MED
        self.DV = DV
        self.IV = sensitive_col[0].upper()

        self.th = th

        self.file_name = self.dataset_name + "_" + str(self.version)

        self.folder_path = os.path.join(self.path, 'DATA_' + str(self.version))
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)

        self.path_splits = os.path.join(self.path, 'splits_data')

        self.data_path = os.path.join(self.folder_path, self.file_name + '.csv')
        if not os.path.exists(self.data_path):
            self.create_data()

    def create_data(self):
        pass

    def read_data(self):
        data = pd.read_csv(self.data_path)
        cols = ['QID', 'UID'] + self.MED + [self.DV, self.IV]
        data = data[cols]
        return data

    def create_splits(self):
        if not os.path.exists(self.path_splits):
            os.makedirs(self.path_splits)
            if 'intersectional' in self.dataset.columns:
                balance_split_col = 'intersectional'
            else:
                balance_split_col = self.sensitive_col
            create_train_split(self.dataset_name, self.dataset, self.query_col, 'cid', balance_split_col,
                               self.score_col, self.path_splits, self.k_fold, 1 - self.ratio_split)

    def get_pos_samples(self, data):
        return data[data['Y'] > self.th]
