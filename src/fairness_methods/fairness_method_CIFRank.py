import os
from pathlib import Path
import pandas as pd

from src.fairness_methods.fairness_method import FairnessMethod
from src.modules.CIFRank_module.generate_counterfactual_data import gen_counterfactual_data_qid

project_dir = Path.cwd()


class CIFRank(FairnessMethod):
    def __init__(self, configs, dataset):
        self.counter_path = None
        self.counter_path_dir = None
        self.causal_path = None
        self.causal_path_dir = None
        super().__init__(configs, dataset)

    def set_paths_specifics(self, out_path):
        self.causal_path_dir = os.path.join(out_path, "parameter_data")
        self.counter_path_dir = os.path.join(out_path, "counterfactual_data")

    def set_paths_runs_specifics(self, i):
        self.causal_path = os.path.join(self.causal_path_dir, str(i))
        self.counter_path = os.path.join(self.counter_path_dir, str(i))

    def generate_fair_data(self, data_train, data_test):
        if not os.path.exists(self.causal_path):
            causal_path_train = os.path.join(self.causal_path, 'train')
            os.makedirs(causal_path_train)
            self.run_causal_model(data_train, causal_path_train)

        if not os.path.exists(self.counter_path):
            causal_path_train = os.path.join(self.causal_path, 'train')
            self.generate_counterfactual_data(data_train, 'count_train.csv', causal_path_train)
            self.generate_counterfactual_data(data_test, 'count_test.csv', causal_path_train)

        counter_data_train = pd.read_csv(os.path.join(self.counter_path, 'count_train.csv'))
        counter_data_test = pd.read_csv(os.path.join(self.counter_path, 'count_test.csv'))

        return counter_data_train, counter_data_test

    def run_causal_model(self, data, causal_path):
        from rpy2 import robjects
        from rpy2.robjects import pandas2ri

        qids = data['QID'].unique()
        for qid in qids:
            temp = data[data['QID'] == qid].copy()
            temp = self.dataset.get_pos_samples(temp)
            try:
                pandas2ri.activate()
                r = robjects.r
                r_script = os.path.join(project_dir, "src/modules/CIFRank_module/R/estimate_causal_model.R")
                r.source(r_script, encoding="utf-8")
                r.estimate_causal_model(temp, self.dataset.IV, self.dataset.DV,
                                        self.dataset.MED, self.configs['METHOD']['control'],
                                        os.path.join(causal_path, str(qid)))
            except:
                if len(os.listdir(causal_path)) != 0:
                    print("Error")
                    df = pd.DataFrame(columns=["Mediators"])
                    df["Mediators"] = 'nan'
                    df.to_csv(os.path.join(causal_path, 'identified_mediators.csv'))

            print("Save med results")
            self.save_med_results(data, os.path.join(causal_path, str(qid)))

    def generate_counterfactual_data(self, data, file_name, causal_path):
        gen_counterfactual_data_qid(data, causal_path, self.counter_path, file_name, self.configs['METHOD'],
                                    self.dataset.IV, self.dataset.DV, self.dataset.MED)

    def save_med_results(self, temp, out_path):
        if os.path.exists(os.path.join(out_path, 'med_output.txt')):
            with open(os.path.join(out_path, 'med_output.txt'), 'r') as f:
                content = f.readlines()

            results_dict = dict()
            next_indirect = False
            for line in content:
                line = line.strip()
                if line.startswith('For the predictor'):
                    if len(results_dict.keys()) == 0:
                        pred = line.split(' ')[3]
                        df_med = pd.DataFrame(columns=['Metric', 'Estimate'])
                        results_dict[pred] = ''
                    else:
                        results_dict[pred] = df_med
                        pred = line.split(' ')[3]
                        df_med = pd.DataFrame(columns=['Metric', 'Estimate'])

                if line.startswith('The estimated total effect:'):
                    total_effect = float(line.split(' ')[4])
                    temp_df = pd.DataFrame([['Total Effect', total_effect]], columns=['Metric', 'Estimate'])
                    df_med = pd.concat([df_med, temp_df], ignore_index=True)

                if next_indirect:
                    splits = line.split(' ')
                    if splits[0] == '':
                        indirect_effect = float(line.split(' ')[1])
                    else:
                        indirect_effect = float(line.split(' ')[0])
                    temp_df = pd.DataFrame([['Indirect Effect', indirect_effect]], columns=['Metric', 'Estimate'])
                    df_med = pd.concat([df_med, temp_df], ignore_index=True)
                    next_indirect = False

                if line.startswith('y1.all'):
                    next_indirect = True

            results_dict[pred] = df_med

            pred_groups = [p.split('pred')[1] for p in results_dict.keys()]
            groups = temp[self.dataset.IV].unique()
            pred_gr = [g for g in groups if g not in pred_groups and g != self.configs['METHOD']['control']][0]
            index = 0
            print(results_dict)
            for key in results_dict.keys():
                index = index + 1
                df_med = results_dict[key]
                direct_effect = df_med[df_med['Metric'] == 'Total Effect']['Estimate'].values[0] - \
                                df_med[df_med['Metric'] == 'Indirect Effect']['Estimate'].values[0]
                temp_df = pd.DataFrame([['Direct Effect', direct_effect]], columns=['Metric', 'Estimate'])
                df_med = pd.concat([df_med, temp_df], ignore_index=True)

                if key == 'pred':
                    file_name = pred_gr + '_med.csv'
                elif 'pred.temp1$x' in key:
                    file_name = groups[index] + '_med.csv'
                else:
                    file_name = key.split('pred')[1] + '_med.csv'

                df_med.to_csv(os.path.join(out_path, file_name))