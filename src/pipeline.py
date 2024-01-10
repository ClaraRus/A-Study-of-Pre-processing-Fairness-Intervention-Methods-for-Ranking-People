import ast
import pandas as pd
from sklearn import preprocessing
import os
from pathlib import Path

from src.evaluate.evaluate import evaluate, evaluate_runs
from src.ranklib.RankLib import RankLib
from src.utils.constants import fairness_method_mapping, dataset_mapping
from src.utils.utils import readFromJson, writeToJson
project_dir = Path.cwd()


class Pipeline:
    def __init__(self, config_path):
        self.trainMethod = None
        self.pred_path_dir = None
        self.eval_path_dir = None
        self.out_path = None
        self.ranklib_path_dir = None
        self.ranklib_path = None
        self.eval_path = None
        self.pred_path = None

        self.configs = readFromJson(config_path)
        self.dataset = dataset_mapping[self.configs['DATA']['name']](self.configs['DATA']['path'],
                                                                     self.configs['DATA']['name'],
                                                                     self.configs['DATA']['version'],
                                                                     self.configs['DATA']['settings'],
                                                                     self.configs['DATA']['MED'],
                                                                     self.configs['DATA']['DV'])

        self.fairness_method = fairness_method_mapping[self.configs['METHOD']['name']](self.configs, self.dataset)
        self.set_paths()

    def set_paths(self):
        self.out_path = os.path.join(self.dataset.folder_path, self.configs['METHOD']['name'],
                                     'EXP_' + str(self.configs['EXP']['num']))

        self.ranklib_path_dir = os.path.join(self.out_path, "ranklib_data")
        self.eval_path_dir = os.path.join(self.out_path, "evaluation_res", self.configs['LTR']['ranker'])
        self.pred_path_dir = os.path.join(self.out_path, "predictions", self.configs['LTR']['ranker'])

        self.fairness_method.set_paths_specifics(self.out_path)

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        writeToJson(os.path.join(self.out_path, "config.json"), self.configs)


    def set_paths_runs(self, i):
        self.ranklib_path = os.path.join(self.ranklib_path_dir, str(i))
        self.pred_path = os.path.join(self.pred_path_dir, str(i))
        self.eval_path = os.path.join(self.eval_path_dir, str(i))
        self.fairness_method.set_paths_runs_specifics(i)

    def read_splits(self, data, i):
        with open(os.path.join(self.dataset.path_splits, 'train_candidates_' + str(i) + '.txt'), 'r') as f:
            content = f.read()
        train_ids = ast.literal_eval(content)
        cids = data['UID'].apply(lambda x: '_'.join(x.split('_')[1:]))
        mask_train = cids.isin(train_ids)

        data_train = data[mask_train]
        data_test = data[~mask_train]

        return data_train, data_test

    def start(self):
        data = self.dataset.read_data()

        for run in range(self.configs['DATA']['settings']['k_fold']):
            data_train, data_test = self.read_splits(data, run)
            data_train, data_test = self.normalize_data(data_train, data_test, run)
            self.set_paths_runs(run)
            self.run(data_train, data_test)

        evaluate_runs(self.configs['EVAL'], self.dataset.folder_path, self.eval_path_dir,
                      self.configs['DATA']['settings']['k_fold'], "test")

    def run(self, data_train, data_test):
        data_train_fair, data_test_fair = self.fairness_method.generate_fair_data(data_train, data_test)

        if self.configs['METHOD']["name"] != 'FAIR':
            self.train_ltr(data_train_fair, data_test_fair)
        self.evaluate(data_test_fair, file_name="prediction.txt")

    def train_ltr(self, data_train, data_test):
        self.trainMethod = RankLib(self.configs, self.dataset, self.ranklib_path, self.pred_path)
        if not os.path.exists(self.ranklib_path):
            self.trainMethod.train_model(data_train, data_test)

    def generate_predictions(self, data_fair, file_name):
        file_path = os.path.join(self.pred_path, "pred.csv")
        if not os.path.exists(file_path):
            predictions = self.trainMethod.generate_predictions(data_fair, file_name)
            if not os.path.exists(self.pred_path):
                os.makedirs(self.pred_path)
            predictions.to_csv(file_path)
        else:
            predictions = pd.read_csv(os.path.join(file_path))

        return predictions

    def evaluate(self, data_fair, file_name):
        predictions = self.generate_predictions(data_fair, file_name)
        path = os.path.join(self.eval_path, 'test')

        if not os.path.exists(path):
            evaluate(predictions, path,
                     self.dataset.IV,
                     self.configs['EVAL'])

    def normalize_data(self, data_train, data_test, i):
        norm_data_path = os.path.join(self.dataset.folder_path, self.dataset.file_name)
        if not os.path.exists(norm_data_path + '_normalized_train' + str(i) + '.csv'):
            min_max_scaler = preprocessing.MinMaxScaler()
            norm_cols = self.dataset.MED.copy()
            norm_cols.append(self.dataset.DV)

            data_train_norm = data_train.copy()
            data_test_norm = data_test.copy()
            for qid in data_train['QID'].unique():
                mask_qid_train = data_train_norm['QID'] == qid
                mask_qid_test = data_test_norm['QID'] == qid
                data_train_norm.loc[mask_qid_train, norm_cols] = min_max_scaler.fit_transform(
                    data_train_norm[mask_qid_train][norm_cols])
                data_test_norm.loc[mask_qid_test, norm_cols] = min_max_scaler.transform(
                    data_test[mask_qid_test][norm_cols])

            data_train_norm.to_csv(norm_data_path + '_normalized_train' + str(i) + '.csv')
            data_test_norm.to_csv(norm_data_path + '_normalized_test' + str(i) + '.csv')

        else:
            data_train_norm = pd.read_csv(norm_data_path + '_normalized_train' + str(i) + '.csv')
            data_test_norm = pd.read_csv(norm_data_path + '_normalized_test' + str(i) + '.csv')

        return data_train_norm, data_test_norm