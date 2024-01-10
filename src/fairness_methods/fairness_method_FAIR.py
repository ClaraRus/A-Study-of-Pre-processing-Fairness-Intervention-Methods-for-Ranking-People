import os
import pandas as pd
from fairsearchcore.models import FairScoreDoc
import fairsearchcore as fsc

from src.fairness_methods.fairness_method import FairnessMethod
from src.ranklib.generate_ranklib_data import assign_judgements


def doc_to_df(docs, data_test):
    predictions = []
    rank = 1
    for doc in docs:
        temp_df = data_test[data_test['UID'] == doc.id]
        temp_df['Y__Y'] = [rank]
        rank = rank + 1
        predictions.append(temp_df)
    return pd.concat(predictions)


class FAIRRanking(FairnessMethod):
    def __init__(self, configs, dataset):
        self.fair_path = None
        self.fair_path_dir = None
        super().__init__(configs, dataset)

    def set_paths_specifics(self, out_path):
        self.fair_path_dir = os.path.join(out_path, "predictions", self.configs['LTR']['ranker'])

    def set_paths_runs_specifics(self, i):
        self.fair_path = os.path.join(self.fair_path_dir, str(i))

    def generate_fair_data(self, data_train, data_test):
        if not os.path.exists(self.fair_path):
            columns = self.dataset.MED + [self.dataset.DV]
            data_test = assign_judgements(data_test, columns, self.configs['LTR']['rel_max'], self.dataset.th)

            def check_is_protected(x, data, k):
                s_value = x[self.dataset.IV][0]
                diversity_k = sum((data.head(k)[self.dataset.IV[0]] == s_value).values)
                th_diversity = k / len(data[self.dataset.IV[0]].unique())

                if diversity_k > th_diversity:
                    return False
                else:
                    return True

            def init_FairDocs(data, k):
                # Extract 'UID' and 'judgement' columns from the group
                docs = []
                for _, x in data.iterrows():
                    docs.append(FairScoreDoc(x['UID'], x['judgement'], check_is_protected(x, data, k)))
                return docs

            p = self.configs['METHOD']['p']  # proportion of protected candidates in the topK elements (value should be between 0.02 and 0.98)
            alpha = self.configs['METHOD']['alpha']  # significance level (value should be between 0.01 and 0.15)
            k = self.configs['METHOD']['k']

            unfair_ranking_qids = data_test.groupby('QID').apply(lambda x: init_FairDocs(x, k))

            # let's check the ranking is considered fair
            re_ranked_list = []
            for unfair_ranking_qid in unfair_ranking_qids:
                # create the Fair object
                fair = fsc.Fair(k, p, alpha)

                # now re-rank the unfair ranking
                re_ranked, rest_of_ranking = fair.re_rank(unfair_ranking_qid)
                re_ranked = re_ranked + rest_of_ranking
                re_ranked_list.append(re_ranked)
            return None, self.format_data(re_ranked_list, data_test)
        else:
            predictions = pd.read_csv(os.path.join(self.fair_path, 'pred.csv'))
            return None, predictions

    def format_data(self, re_ranked_list, data_test):
        predictions = []

        for ranked_list in re_ranked_list:
            predictions.append(doc_to_df(ranked_list, data_test))
        predictions = pd.concat(predictions)
        if not os.path.exists(self.fair_path):
            os.makedirs(self.fair_path)
        predictions.to_csv(os.path.join(self.fair_path, 'pred.csv'))
        return predictions