import numpy as np
import pandas as pd
import math
import os

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise
from src.utils.utils import get_quotas_count, get_sort_df, writeToCSV, readFromJson


def evaluate_runs(args_eval, path_data, eval_path, runs, split):
    qids = readFromJson(os.path.join(path_data, 'qids.json'))

    for qid in qids.values():
        for eval_measure in args_eval['measures']:
            output_f = os.path.join(eval_path, "Eval_QID_" + str(qid) + '_' + eval_measure + ".csv")

            if not os.path.exists(output_f):
                res_all = []
                for r in range(runs):
                    file_name = 'Eval_QID_' + str(qid) + '_' + eval_measure + '.csv'
                    path = os.path.join(eval_path, str(r), split, eval_measure, file_name)

                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        res_all.append(df)

                res_all = pd.concat(res_all)
                res_all = res_all.groupby(['rank', 'group','k']).mean().reset_index()
                writeToCSV(output_f, res_all)
                print("--- Save eval file in ", output_f, " --- \n")


def evaluate(df, eval_path, IV, args_eval):
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    qids = df['QID'].unique()
    sensitive_groups = df[IV].unique()

    for eval_measure in args_eval['measures']:
        if not os.path.exists(os.path.join(eval_path, eval_measure)):
            os.makedirs(os.path.join(eval_path, eval_measure))
            for qid in qids:
                df_qid = df[df['QID'] == qid]
                res_qid = evaluate_qid(df_qid, eval_measure, IV, sensitive_groups, args_eval)

                output_f = os.path.join(eval_path, eval_measure, "Eval_QID_" + str(qid) + "_" + eval_measure + ".csv")
                writeToCSV(output_f, res_qid)
                print("--- Save eval file in ", output_f, " --- \n")


def evaluate_qid(df, eval_measure, IV, sensitive_groups, args_eval):
    EVAL_RANKINGS = args_eval['rankings']

    res_df = pd.DataFrame(columns=["run", "rank", "k", "group", eval_measure])
    k_list = args_eval['k_list']

    for ranking in EVAL_RANKINGS:
        for ki in k_list:

            res_row = [1, ranking, ki]
            all_row = res_row + ["all"]

            if eval_measure == "diversity":
                all_row.append(1)

            sort_df = get_sort_df(ranking, df, ki)

            if 'individual_fairness' in eval_measure:
                if "__" in ranking:
                    yNN = compute_individual_fairness(df, ranking)
                    all_row.append(yNN)
                else:
                    all_row.append(-1)

            if len(sort_df) < ki:
                sort_df["rank"] = list(range(1, len(sort_df) + 1))
            else:
                sort_df["rank"] = list(range(1, ki + 1))

            if 'NDCG' in eval_measure:
                ndcg = calculate_ndcg(df, ranking, ki)
                all_row.append(ndcg)

            res_df.loc[res_df.shape[0]] = all_row

            # group-level evaluation
            cur_quotas = get_quotas_count(sort_df, IV, sensitive_groups)
            for gi in sensitive_groups:
                gi_row = res_row + [gi]

                if eval_measure == "diversity":
                    if gi in cur_quotas:
                        gi_row.append(cur_quotas[gi])
                    else:
                        gi_row.append(0)

                if 'individual_fairness' in eval_measure:
                    gi_row.append(-1)

                if 'NDCG' in eval_measure:
                    gi_row.append(-1)

                res_df.loc[res_df.shape[0]] = gi_row
    return res_df


def compute_individual_fairness(data, ranking, weights=None):
    feature_columns = [col for col in data if 'X' in col and '_' not in col]

    if weights is None:
        distances_data = pairwise.euclidean_distances(data[feature_columns].to_numpy(),
                                                      data[feature_columns].to_numpy())
    else:
        distances_data = pdist(data[feature_columns].to_numpy(), 'minkowski', p=2, w=weights)
        distances_data = squareform(distances_data)
    exposers = data[ranking].apply(lambda x: 1 / math.log2(x + 1))
    distances_exposer = pairwise.euclidean_distances(exposers.to_numpy().reshape(-1, 1),
                                                     exposers.to_numpy().reshape(-1, 1))

    yNN = 1 - np.mean(np.abs(distances_data - distances_exposer))
    return yNN


def calculate_ndcg(data, ranking, k):
    dcg = calculate_dcg(data[ranking], k)

    ideal_ranking = ranking.split('__')[0]
    ideal_dcg = calculate_dcg(data[ideal_ranking], k)

    return dcg / ideal_dcg


def calculate_dcg(judgements, k):
    dcg = 0
    for i in range(0, k):
        dcg = dcg + judgements.iloc[i] / math.log(i + 2, 2)
    return dcg