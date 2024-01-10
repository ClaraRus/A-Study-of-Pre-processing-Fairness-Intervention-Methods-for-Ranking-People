import json
import os.path
import pickle
import numpy as np
import pandas as pd
from nltk import tokenize

from datasets.dataset import Dataset
from src.utils.utils import clean_text, extract_pronouns, add_vector_representation, compute_cosine_similarity

West_EU = ['English', 'Welsh', 'British', 'Irish', 'Basque', 'Faroese', 'Belgian', 'Dutch', 'Austrian', 'Norwegian',
           'German', 'French', 'Italian', 'Portuguese', 'Catalan']
East_EU = ['Ukrainian', 'Russian', 'Bulgarian', 'Greek', 'Estonian', 'Czech', 'Romanian', 'Hungarian', 'Albanian',
           'Botswana', 'Moldovan', 'Belarusian', 'Georgian', 'Slovak', 'Macedonian', 'Lithuanian', 'Latvian',
           'Montenegrin']
European = West_EU + East_EU

Western_Asian = ['Israeli', 'Pakistani', 'Afghan', 'Iranian', 'Palestinian', 'Azerbaijani', 'Jordanian', 'Syrian',
                 'Armenian', 'Iraqi', 'Lebanese', 'Bahraini', 'Kuwaiti', 'Cypriot', 'Qatari', 'Saudi', 'Omani']
South_Asians = ['Indian', 'Tamil', 'Nepalese', 'Bangladeshi', 'Bhutanese', 'Maldivian']
Southeast_Asians = ['Burmese', 'Thai', 'Taiwanese', 'Malaysian', 'Vietnamese', 'Indonesian', 'Filipino', 'Singaporean',
                    'Cambodian', 'Tibetan', 'Yemeni']
East_Asians = ['Chinese', 'Japanese', 'Korean']
Central_Asians = ['Mongolian']
Asian = Western_Asian + South_Asians + Southeast_Asians + East_Asians + Central_Asians

Latin_American = ['Peruvian', 'Uruguayan', 'Venezuelan', 'Bolivian', 'Mexican', 'Jamaican', 'Argentine', 'Brazilian',
                  'Bahamian', 'Guatemalan', 'Colombian', 'Samoan', 'Cuban', 'Chilean']
African = ['Angolan', 'Nigerien', 'Beninese', 'Liberian', 'Burundian', 'Swazi', 'Togolese', 'Libyan', 'Ugandan',
           'Emirati', 'Egyptian', 'Tunisian', 'Kenyan', 'Gambian', 'Algerian', 'Nigerian',
           'Ethiopian', 'Moroccan', 'Sudanese', 'Tanzanian', 'Cameroonian', 'Senegalese', 'Zambian', 'Ghanaian',
           'Guinean', 'Malagasy', 'Rwandan', 'Equatoguinean', 'Burkinabé', 'Malian']
Non_European = Asian + Latin_American + African
nationality_group = {"Asian": Asian, "African": African, "Latin_Americans": Latin_American, "West-European": West_EU,
                     "East-European": East_EU}


def count_tf(text, title):
    count = 0
    for t in title.split('_'):
        count = count + text.count(t)
    return count


def check_name(name, clean_raw):
    from src.utils.constants import word2vec_google_news

    names = name.lower().strip().split(" ")
    return [n != '' and len(n) > 2 and n in word2vec_google_news and n in clean_raw for n in names]


class BIOSDataset(Dataset):
    def __init__(self, path, dataset_name, version, settings=None, MED=None, DV=None):
        self.dataset = None
        self.settings = settings
        self.query_col = 'title'
        self.sensitive_col = 'gender'
        self.score_col = 'cosine_similarity'
        self.features_col = ['TF', 'len', 'words']
        self.th = 0.5
        super().__init__(path, dataset_name, query_col=self.query_col, sensitive_col=self.sensitive_col,
                         score_col=self.score_col, features_col=self.features_col, DV=DV,
                         MED=MED, k_fold=self.settings['k_fold'], ratio_split=self.settings['ratio_split'],
                         version=version, th=self.th)

    def create_data(self):
        pkl_path = os.path.join(self.path, self.dataset_name + '.pkl')
        if not os.path.exists(pkl_path):
            self.concat_data(pkl_path)

        pkl_path_cid = os.path.join(self.path, self.dataset_name + '_candidates.pkl')
        if not os.path.exists(pkl_path_cid):
            self.dataset = pd.DataFrame.from_dict(pd.read_pickle(pkl_path))
            self.create_candidates_id(pkl_path_cid)
        else:
            self.dataset = pd.DataFrame.from_dict(pd.read_pickle(pkl_path_cid))

        pkl_path_features = os.path.join(self.path, self.dataset_name + '_candidates_features.pkl')
        if not os.path.exists(pkl_path_features):
            self.dataset = pd.DataFrame.from_dict(pd.read_pickle(pkl_path_cid))
            self.set_nationality()
            self.set_intersectional()
            print("Clean data...")
            self.clean_data()

            print("Compute features...")
            self.extract_features()
            self.dataset.to_pickle(pkl_path_features)
        else:
            self.dataset = pd.DataFrame.from_dict(pd.read_pickle(pkl_path_features))

        self.create_splits()

        print("Format Data...")
        self.format_data()

        print("Save Format Data...")
        self.dataset.to_csv(self.data_path)

    def clean_data(self):
        # drop duplicates by name
        mask_drop = self.dataset.name.duplicated()
        self.dataset = self.dataset[~mask_drop]

        self.dataset['clean_raw'] = self.dataset.apply(lambda row: clean_text(row['raw']), axis=1)
        mask_drop = np.logical_or(self.dataset['raw'] == '', self.dataset['clean_raw'] == '')
        self.dataset = self.dataset[~mask_drop]

        # remove candidates that do not have names in w2vec or in text, or no pronouns in text
        self.dataset['pronouns'] = self.dataset['clean_raw'].apply(lambda x: extract_pronouns(x))
        mask_keep = self.dataset[['name', 'clean_raw', 'pronouns']].apply(
            lambda x: sum(check_name(x['name'], x['clean_raw'])) != 0 and len(x['pronouns']) > 0, axis=1)
        self.dataset = self.dataset[mask_keep]

    def format_data(self):
        cols = ['QID', 'UID'] + self.MED + [self.DV, self.IV]
        data = pd.DataFrame(columns=cols)

        # save encoding of occupations to qids
        queries = self.dataset[self.query_col].unique()
        queries_dict = {queries[x]: x for x in range(len(queries))}
        with open(os.path.join(self.folder_path, "qids.json"), "w") as fp:
            json.dump(queries_dict, fp)

        # set candidate ids
        with open(os.path.join(self.path, 'candidates_ids.pkl'), 'rb') as f:
            candidates_ids = pickle.load(f)
        cids = self.dataset['name'].apply(lambda x: candidates_ids[x])

        data.QID = self.dataset[self.query_col].apply(lambda x: str(queries_dict[x]))
        data.UID = data.QID + '_' + cids

        data[self.IV] = self.dataset[self.sensitive_col]
        data.Y = self.dataset['score']
        data[self.MED] = self.dataset[self.features_col]

        new_cols = {k: 'X' + str(v) for v, k in enumerate(self.features_col)}
        with open(os.path.join(self.folder_path, "features.json"), "w") as fp:
            json.dump(new_cols, fp)

        self.MED = list(new_cols.values())
        data = data.rename(columns=new_cols)

        self.dataset = data

    def extract_features(self):
        from src.utils.constants import word2vec_google_news

        self.dataset['len'] = self.dataset['clean_raw'].apply(lambda x: len(x))
        self.dataset['words'] = self.dataset['clean_raw'].apply(lambda x: len(tokenize.word_tokenize(x)))
        self.dataset['TF'] = self.dataset.apply(lambda row: count_tf(row['raw'].lower(), row[self.query_col]), axis=1)

        self.dataset = add_vector_representation(self.dataset, 'title', word2vec_google_news)
        self.dataset = add_vector_representation(self.dataset, 'clean_raw', word2vec_google_news)
        self.dataset['score'] = compute_cosine_similarity(self.dataset, 'vec_title', 'vec_clean_raw')

    def create_candidates_id(self, pkl_path):
        self.dataset['name'] = self.dataset.name.apply(lambda x: " ".join(x))
        if not os.path.exists(os.path.join(self.path, 'candidates_ids.pkl')):
            self.dataset['cid'] = self.dataset.groupby(['name'], sort=False).ngroup() + 1
            self.dataset['cid'] = self.dataset.cid.apply(lambda x: 'candidate_' + str(x))

            candidates_ids = dict(zip(self.dataset.name, self.dataset.cid))

            with open(os.path.join(self.path, 'candidates_ids.pkl'), 'wb') as f:
                pickle.dump(candidates_ids, f)
        else:
            with open(os.path.join(self.path, 'candidates_ids.pkl'), 'rb') as f:
                candidates_ids = pickle.load(f)
                mask = self.dataset.name.apply(lambda x: x in candidates_ids.keys())
                self.dataset = self.dataset[mask]
                self.dataset['cid'] = self.dataset.name.apply(lambda x: str(candidates_ids[x]))
        self.dataset.to_pickle(pkl_path)

    def concat_data(self, pkl_path):
        data = []
        dir_path = os.path.join(self.path, 'biosbias-master', 'no_failed_paths')
        files = os.listdir(dir_path)
        for f in files:
            if f.startswith('CC-MAIN') and f.endswith('.pkl'):
                temp = pd.DataFrame.from_dict(pd.read_pickle(os.path.join(dir_path, f)))
                data.append(temp)
        self.dataset = pd.concat(data)

        self.dataset.name = self.dataset.name.apply(lambda x: " ".join(x))

        self.dataset.to_pickle(pkl_path)

    def set_nationality(self):
        from name2nat import Name2nat
        nat_detector = Name2nat()
        self.dataset['nationality'] = self.dataset.name.apply(lambda x: nat_detector(x)[0][1][0][0])
        deleted = ['American', 'Canadian', 'Australian']  # due to ambiguity
        mask_keep = self.dataset['nationality'].apply(
            lambda x: x not in deleted)
        self.dataset = self.dataset[mask_keep]

        self.dataset['nationality'] = self.dataset['nationality'].apply(
            lambda x: assign_nat_group(x, nationality_group))

        mask_keep = self.dataset['nationality'].apply(
            lambda x: x != 'null')
        self.dataset = self.dataset[mask_keep]

    def set_intersectional(self):
        self.dataset['intersectional'] = self.dataset['gender'] + self.dataset['nationality']


def assign_nat_group(nationality, nationality_groups):
    for group in nationality_groups.items():
        if nationality in group[1]:
            return group[0]
    return "null"
