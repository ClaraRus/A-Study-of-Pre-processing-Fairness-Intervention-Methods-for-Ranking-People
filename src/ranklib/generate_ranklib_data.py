import pandas as pd


def create_ranklib_data(df_copy, cols_train, DV):
    idx = 0
    cols_keep = []

    for ci in cols_train:
        cols_keep.append(ci)
        df_copy[ci] = df_copy[ci].apply(lambda x: str(idx + 1) + ":" + str(round(x, 4)))
        idx = idx + 1

    df_copy["QID"] = df_copy["QID"].apply(lambda x: "qid:" + str(x))
    df_copy["UID"] = df_copy[["UID", "judgement", DV]].astype(str).apply(
        lambda x: ("#docid={};rel={};" + DV + "={};").format(x.iloc[0], x.iloc[1], x.iloc[2]),
        axis=1)

    # shuffle df test
    groups = [df_copy for _, df_copy in df_copy.groupby('QID')]
    shuffled_groups = [gr.sample(frac=1) for gr in groups]
    df_copy = pd.concat(shuffled_groups).reset_index(drop=True)

    df_copy = df_copy[["judgement", "QID"] + cols_keep + ["UID"]]

    return df_copy


def add_judgement(x, rel_max, th):
    mask_pos = x['Y'].apply(lambda x: round(x, 2) > th)

    pos_x = x[mask_pos]
    pos_x['judgement'] = x[mask_pos]['Y'].rank(ascending=True, method='dense')

    min_rank = pos_x['judgement'].min()
    max_rank = pos_x['judgement'].max()
    pos_x['judgement'] = ((pos_x['judgement'] - min_rank) / (max_rank - min_rank + 1)) * rel_max
    pos_x['judgement'] = pos_x['judgement'].round().astype(int)

    if len(pos_x) > 0:
        if max(pos_x['judgement']) < rel_max:
            diff = rel_max - max(pos_x['judgement'])
            pos_x['judgement'] = pos_x['judgement'].apply(lambda x: x + diff)
        if min(pos_x['judgement']) < 1:
            mask = pos_x['judgement'] < 1
            pos_x['judgement'][mask] = 1

    neg_x = x[~mask_pos]
    neg_x['judgement'] = 0
    x = pd.concat([pos_x, neg_x])

    return x


def assign_judgements(df, cols, rel_max, th):
    df = df.groupby(["QID"]).apply(lambda x: x.sort_values(by=cols[-1], ascending=False)).reset_index(
        drop=True)
    temp = df.groupby(["QID"]).apply(
        lambda x: add_judgement(x, rel_max, th))
    df = temp.reset_index(drop=True)

    return df