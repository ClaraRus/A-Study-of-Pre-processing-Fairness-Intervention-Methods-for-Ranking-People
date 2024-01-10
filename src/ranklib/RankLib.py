import os
import shutil
import subprocess
from pathlib import Path
import pandas as pd

from src.ranklib.generate_predictions import get_LTR_predict
from src.ranklib.generate_ranklib_data import create_ranklib_data, assign_judgements
from src.utils.utils import get_col_map, writeToTXT, check_nan

project_dir = Path.cwd()


class RankLib():
    def __init__(self, configs, dataset, ranklib_path, pred_path):
        self.name = 'ranklib'
        self.configs = configs
        self.dataset = dataset
        self.ranklib_path = ranklib_path
        self.pred_path = pred_path

    def train_model(self, data_train, data_test):
        self.generate_ranklib_data(data_train, data_test)
        args = [os.path.join(project_dir, "src", "ranklib"), self.configs['LTR']['metric'],
                self.configs['LTR']['top_k'], self.configs['LTR']['rel_max'],
                self.configs['LTR']['ranker'], self.configs['LTR']['ranker_id'],
                self.configs['LTR']['lr'], self.configs['LTR']['epochs']]

        experiments = [f for f in os.listdir(self.ranklib_path) if "__" in f]
        train_sets = set([exp.split('__')[0] for exp in experiments])
        test_sets = set([exp.split('__')[1] for exp in experiments])

        models = dict()
        for train_set in train_sets:
            exp = [e for e in experiments if train_set in e][0]

            if not os.path.exists(
                    os.path.join(self.ranklib_path, exp, 'ranklib-experiments', self.configs['LTR']['ranker'])):
                subprocess.check_call(
                    [os.path.join(project_dir, "src", "ranklib", "run-LTR-model.sh"), args[0], str(args[1]),
                     str(args[2]), str(args[3]), str(args[4]), str(args[5]), os.path.join(self.ranklib_path, exp),
                     str(args[6]),
                     str(args[7]), "none"])
            exp_dir = \
                os.listdir(os.path.join(self.ranklib_path, exp, 'ranklib-experiments', self.configs['LTR']['ranker']))[
                    0]
            models[train_set] = os.path.join(
                os.path.join(self.ranklib_path, exp, 'ranklib-experiments', self.configs['LTR']['ranker']), exp_dir)

        for exp in experiments:
            train_set = exp.split("__")[0]
            test_set = exp.split("__")[1]
            model = models[train_set]
            ranklib_exp_path = os.path.join(self.ranklib_path, exp, 'ranklib-experiments',
                                            self.configs['LTR']['ranker'])
            if not os.path.exists(ranklib_exp_path):
                # os.makedirs(ranklib_exp_path)
                exp_dir = model.split('/')[-1]
                shutil.copytree(model, os.path.join(ranklib_exp_path, exp_dir))
                os.remove(os.path.join(ranklib_exp_path, exp_dir, 'R', 'predictions', 'prediction.txt'))

            exp_dir = [f for f in os.listdir(ranklib_exp_path) if "experiments" in f][0]
            exp_path = os.path.join(self.ranklib_path, exp, 'ranklib-experiments', self.configs['LTR']['ranker'],
                                    exp_dir)
            if not os.path.exists(os.path.join(exp_path, 'R', 'predictions', 'prediction.txt')):
                subprocess.check_call(
                    [os.path.join(project_dir, "src", "ranklib", "run-LTR-model.sh"), args[0], str(args[1]),
                     str(args[2]), str(args[3]), str(args[4]), str(args[5]), os.path.join(self.ranklib_path, exp),
                     str(args[6]), str(args[7]), exp_path])

    def generate_predictions(self, data_fair, file_name):
        if not os.path.exists(os.path.join(self.pred_path, "pred.csv")):
            predictions = get_LTR_predict(data_fair, self.ranklib_path, self.configs['LTR']['ranker'],
                                          self.pred_path,
                                          self.dataset.DV, self.dataset.MED, file_name)
        else:
            predictions = pd.read_csv(os.path.join(self.pred_path, "pred.csv"))
        return predictions

    def generate_ranklib_data(self, data_train, data_test):

        if 'Unnamed: 0' in data_train:
            data_train = data_train.drop(['Unnamed: 0'], axis=1)

        if 'Unnamed: 0' in data_test:
            data_test = data_test.drop(['Unnamed: 0'], axis=1)

        col_map = get_col_map(self.dataset.MED, self.dataset.DV)
        experiments = [(self.configs['LTR']['train_data'][i], self.configs['LTR']['test_data'][i]) for i in
                       range(len(self.configs['LTR']['train_data']))]

        for experiment in experiments:
            if experiment[0] in col_map and experiment[1] in col_map:
                out_dir = os.path.join(self.ranklib_path, experiment[0] + '__' + experiment[1])
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                    cols_train = col_map[experiment[0]]
                    cols_test = col_map[experiment[1]]

                    if check_nan(data_train, cols_train):
                        raise ValueError('Nan values in train!')
                    if check_nan(data_test, cols_test):
                        raise ValueError('Nan values in test!')

                    df_train = data_train.copy()
                    df_train = assign_judgements(df_train, cols_train, self.configs['LTR']['rel_max'], self.dataset.th)

                    cols_train = cols_train[:-1]  # do not use Y as training feature

                    df_train_ranklib = create_ranklib_data(df_train, cols_train, self.dataset.DV)

                    df_test = data_test.copy()
                    df_test = assign_judgements(df_test, cols_test, self.configs['LTR']['rel_max'], self.dataset.th)
                    cols_test = cols_test[:-1]  # do not use Y as training feature
                    df_test_ranklib = create_ranklib_data(df_test, cols_test, self.dataset.DV)

                    output_f = os.path.join(out_dir, "R_train_ranklib.txt")
                    writeToTXT(output_f, df_train_ranklib)

                    output_f = os.path.join(out_dir, "R_test_ranklib.txt")
                    writeToTXT(output_f, df_test_ranklib)
                    print("--- Save ranklib data in", output_f, " --- \n")