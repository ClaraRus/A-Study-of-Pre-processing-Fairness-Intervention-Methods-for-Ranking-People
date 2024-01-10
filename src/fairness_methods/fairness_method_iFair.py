import os
import pandas as pd
from flatbuffers.builder import np

from src.fairness_methods.fairness_method import FairnessMethod
from src.utils.utils import writeToCSV


class iFairRanking(FairnessMethod):
    def __init__(self, configs, dataset):
        self.ifair_path = None
        self.ifair_path_dir = None
        super().__init__(configs, dataset)

    def set_paths_specifics(self, out_path):
        self.ifair_path_dir = os.path.join(out_path, "iFair_data")

    def set_paths_runs_specifics(self, i):
        self.ifair_path = os.path.join(self.ifair_path_dir, str(i))

    def generate_fair_data(self, data_train, data_test):
        from src.modules.iFair_module.iFair import iFair

        if not os.path.exists(self.ifair_path):
            os.makedirs(self.ifair_path)

        if not os.path.exists(os.path.join(self.ifair_path, 'iFair_train_data.csv')):
            distances_path = os.path.join(self.ifair_path, 'distances')
            model = iFair(distances_path, k=self.configs['METHOD']['k'], A_x=self.configs['METHOD']['A_x'],
                          A_z=self.configs['METHOD']['A_z'], max_iter=self.configs['METHOD']['max_iter'],
                          nb_restarts=self.configs['METHOD']['nb_restarts'])

            features_cols = self.dataset.MED.copy()
            features_cols.append(self.dataset.DV)
            nonsensitive_column_indices = list(range(0, data_train[features_cols].shape[1]))

            features_cols.append(self.dataset.IV + '_coded')

            data_train_fair = data_train.copy()
            data_test_fair = data_test.copy()

            codes, uniques = pd.factorize(data_train_fair[self.dataset.IV])
            data_train_fair[self.dataset.IV + '_coded'] = codes

            codes, uniques = pd.factorize(data_test_fair[self.dataset.IV])
            data_test_fair[self.dataset.IV + '_coded'] = codes

            run = self.ifair_path.split('/')[-1]

            if 'QID' not in self.configs['METHOD']:
                self.fit_model(self, model, data_train, features_cols, nonsensitive_column_indices, run)

                data_train_fair = model.transform(data_train_fair[features_cols].to_numpy())
                data_test_fair = model.transform(data_test_fair[features_cols].to_numpy())

            else:
                data_train_fair_list = []
                data_test_fair_list = []
                qids = data_train_fair['QID'].unique()
                for qid in qids:
                    df_qid_train = data_train_fair[data_train_fair['QID'] == qid]
                    df_qid_test = data_test_fair[data_test_fair['QID'] == qid]
                    df_pos_train = self.dataset.get_pos_samples(df_qid_train)

                    if not os.path.exists(os.path.join(self.ifair_path, str(qid) + '__model_parmas.npy')):
                        self.fit_model(model, df_pos_train, features_cols, nonsensitive_column_indices, run, qid)

                    data_train_transformed = model.transform(df_qid_train[features_cols].to_numpy())
                    data_test_transformed = model.transform(df_qid_test[features_cols].to_numpy())
                    data_train_fair_list.append(data_train_transformed)
                    data_test_fair_list.append(data_test_transformed)

                features_cols_fair = [col + '_fair' for col in features_cols]

                train_transformed = np.vstack(data_train_fair_list)
                test_transformed = np.vstack(data_test_fair_list)
                for index, col in enumerate(features_cols_fair):
                    data_train_fair.loc[:, col] = train_transformed[:, index]
                    data_test_fair.loc[:, col] = test_transformed[:, index]

            writeToCSV(os.path.join(self.ifair_path, 'iFair_train_data.csv'), data_train_fair)
            writeToCSV(os.path.join(self.ifair_path, 'iFair_test_data.csv'), data_test_fair)
        else:
            data_train_fair = pd.read_csv(os.path.join(self.ifair_path, 'iFair_train_data.csv'))
            data_test_fair = pd.read_csv(os.path.join(self.ifair_path, 'iFair_test_data.csv'))
        return data_train_fair, data_test_fair

    def fit_model(self, model, data_train, features_cols, nonsensitive_column_indices, run, qid=None):
        if qid is not None:
            out_path = os.path.join(self.ifair_path, str(qid) + '__model_parmas.npy')
        else:
            out_path = os.path.join(self.ifair_path, 'model_parmas.npy')

        if os.path.exists(out_path):
            with open(out_path, 'rb') as f:
                model.opt_params = np.load(f)
        else:
            temp = self.dataset.get_pos_samples(data_train)
            model.fit(temp[features_cols].to_numpy(), run, qid,
                      batch_size=self.configs['METHOD']['batch'],
                      nonsensitive_column_indices=nonsensitive_column_indices)
            with open(out_path, 'wb') as f:
                np.save(f, model.opt_params)