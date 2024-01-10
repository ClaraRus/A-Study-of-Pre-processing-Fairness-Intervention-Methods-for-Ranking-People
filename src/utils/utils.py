import pickle
import random
import numpy as np
import pandas as pd
import re
import os, pathlib, json
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk import tokenize, word_tokenize
from sklearn.metrics import pairwise

lemmatizer = WordNetLemmatizer()

from nltk.stem import PorterStemmer

ps = PorterStemmer()

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def get_col_map(med_cols, score_col):
    ltr_cols = dict()
    original_features = med_cols + [score_col]
    fair_features_cols = [col + '_fair' for col in med_cols]
    ltr_cols["bias"] = original_features
    ltr_cols["fair"] = fair_features_cols + [score_col + '_fair']

    return ltr_cols


def get_quotas_count(_df, IV, sensitive_groups):
    res_dict = {}

    for s in sensitive_groups:
        mask = _df[IV] == s
        res_dict[s] = sum(mask) / len(_df)

    return res_dict


def vec_word2vec(text, word_vectors):
    list_of_words = [word.lower() for word in word_tokenize(text)]
    if (len(list_of_words)) > 0:
        vec_sum = np.zeros(word_vectors[0].shape)
        n = False
        for word in list_of_words:
            if word in word_vectors:
                vec_sum += word_vectors[word]
                n = np.linalg.norm(vec_sum)

        return vec_sum / n if n else vec_sum
    return None


def unit_vector(vec):
    """
    Returns unit vector
    """
    return vec / np.linalg.norm(vec)


def cos_sim(v1, v2):
    """
    Returns cosine of the angle between two vectors
    """

    if v1 is None or v2 is None:
        return -1

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)


def add_vector_representation(data_, column, word2vec):
    data_['vec_' + column] = data_[column].apply(lambda x: vec_word2vec(x.replace("_", " "), word2vec))
    return data_


def compute_cosine_similarity(data_, column1, column2):
    print("Compute cosine similarity...")
    return data_.apply(lambda row: cos_sim(row[column1], row[column2]), axis=1)


def extract_pronouns(x):
    F_pronouns = ['she', 'her', 'hers', 'herself']
    M_pronouns = ['he', 'him', 'his', 'himself']

    words = tokenize.word_tokenize(x.lower())
    pronouns = []
    for word in words:
        if word in F_pronouns or word in M_pronouns:
            pronouns.append(word)
    return " ".join(pronouns)


def clean_text(text, email_=False, alpha_=True, stopwords_=True, stem_=False, lemmantization_=True):
    # all lower
    text = text.lower()

    if email_:
        # remove email header
        text = "\n".join([line for line in text.split('\n') if not '=\"' in line])

    if alpha_:
        # remove all but not words
        text = re.sub('[^a-zA-Z ]+', ' ', text)

    if stopwords_:
        # remove stop-words
        F_pronouns = ['she', 'her', 'hers', 'herself']
        M_pronouns = ['he', 'him', 'his', 'himself']
        stop_words_ = [word for word in stop_words if word not in F_pronouns and word not in M_pronouns]
        words = text.split(' ')
        text = " ".join([word for word in words if word not in stop_words_])

    if lemmantization_:
        # lemmantization
        words = text.split(' ')
        text = " ".join([lemmatizer.lemmatize(word) for word in words])

    if stem_:
        # stemming
        words = text.split(' ')
        text = " ".join([ps.stem(word) for word in words])

    return text


def writeToTXT(file_name_with_path, _df):
    # try:
    #     _df.to_csv(file_name_with_path, header=False, index=False, sep=' ')
    # except FileNotFoundError:
    directory = os.path.dirname(file_name_with_path)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    print("Make folder ", directory)
    _df.to_csv(file_name_with_path, header=False, index=False, sep=' ')


def writeToCSV(file_name_with_path, _df):
    try:
        _df.to_csv(file_name_with_path, index=False)
    except FileNotFoundError:

        directory = os.path.dirname(file_name_with_path)
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print("Make folder ", directory)
        _df.to_csv(file_name_with_path, index=False)


def writeToJson(file_name_with_path, _data):
    directory = os.path.dirname(file_name_with_path)
    if not os.path.exists(directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print("Make folder ", directory)
    with open(file_name_with_path, 'w') as fp:
        json.dump(_data, fp, indent=2)


def readFromJson(file_name_with_path, return_key=None):
    with open(file_name_with_path) as json_data:
        d = json.load(json_data)
    if return_key:
        return d[return_key]
    else:
        return d


def get_sort_df(_sort_col, _df, _k):
    if "__" in _sort_col:
        sort_df = _df.sort_values(by=_sort_col, ascending=True).head(_k)
    else:
        sort_df = _df.sort_values(by=_sort_col, ascending=False).head(_k)
    return sort_df


def select_balance_df(_df, query_col, id_col, group_col, _sort_col, return_ratio=0.7):
    queries = _df[query_col].unique()
    train_queries = []
    drop_qid = []
    for qid in queries:
        pos_df = _df[_df[query_col] == qid]
        groups_pos = pos_df[group_col].unique()
        for g in groups_pos:
            mask_g = pos_df[group_col] == g
            if len(mask_g) > 1:
                uids = pos_df[mask_g][id_col].values
                if len(uids) > 0:
                    ratio = return_ratio * len(uids)
                    train_split = random.sample(list(uids), int(ratio))
                    train_queries.extend(train_split)
            else:
                drop_qid.append(qid)

    # drop qids for which we can't do a proper split between the sensitive groups
    train_queries = [uid for uid in train_queries if uid.split('_')[0] not in drop_qid]
    return train_queries


def create_train_split(data_name, df, query_col, id_col, group_col, _sort_col, data_path, k_fold, ratio_split):
    for i in range(k_fold):
        train_ids = select_balance_df(data_name, df, query_col, id_col, group_col, _sort_col, ratio_split)
        test_ids = set(df[id_col]).difference(train_ids)
        with open(os.path.join(data_path, 'test_candidates_' + str(i) + '.txt'), 'w') as f:
            f.write(str(test_ids))
        with open(os.path.join(data_path, 'train_candidates_' + str(i) + '.txt'), 'w') as f:
            f.write(str(train_ids))


def check_nan(df, cols_train):
    for col in cols_train:
        if df[col].isnull().values.any():
            return True
    return False


def compute_euclidean_distances(X, euclidean_dist_dir, batch, nonsensitive_column_indices):
    if not nonsensitive_column_indices:
        nonsensitive_column_indices = list(range(0, X.shape[1] - 1))

    if not os.path.exists(euclidean_dist_dir):
        os.makedirs(euclidean_dist_dir)
        if batch is not None:
            for i in range(0, len(X), batch):
                D_X_F = pairwise.euclidean_distances(X[i:i + batch, nonsensitive_column_indices],
                                                     X[:, nonsensitive_column_indices])
                with open(os.path.join(euclidean_dist_dir,
                                       'euclidean_distance_' + str(i) + '_' + str(i + batch)) + '.pkl',
                          'wb') as f:
                    pickle.dump(D_X_F, f)
        else:
            D_X_F = pairwise.euclidean_distances(X[:, nonsensitive_column_indices],
                                                 X[:, nonsensitive_column_indices])
            with open(os.path.join(euclidean_dist_dir, 'euclidean_distance.pkl'), 'wb') as f:
                pickle.dump(D_X_F, f)
        return D_X_F