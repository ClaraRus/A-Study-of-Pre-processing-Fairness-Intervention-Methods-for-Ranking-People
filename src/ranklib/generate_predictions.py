import os
from src.utils.utils import writeToCSV, get_col_map


def get_prediction_scores(file_path, run_timer, ranker, file_name='predictions.txt'):
    # return a dict in which key is the uuid and value is their prediction score
    # score is used to retrieve the relative order, not for other use!!!
    pred_latest_path = file_path + "/ranklib-experiments/" + ranker + "/"
    # retrieve the latest experiment folder
    sub_exp = [x for x in os.listdir(pred_latest_path) if "experiments_" in x]
    exp_suffix = max([os.path.join(pred_latest_path, d) for d in sub_exp], key=os.path.getmtime)[-15:]
    pred_latest_path = pred_latest_path + "experiments_" + exp_suffix + "/" + run_timer + "/predictions/" + file_name
    if os.path.exists(pred_latest_path):
        print("**** Reading pred at", pred_latest_path)
        with open(pred_latest_path, "r") as text_file:
            ranker_lines = text_file.read().splitlines()

        preds = [(li.split(" ")[2][0:li.split(" ")[2].find(";rel=")].replace("docid=", ""), int(li.split(" ")[3])) for
                 li in
                 ranker_lines]
        ranker_pred = dict(preds)
        return ranker_pred
    else:
        print("No prediction found for ", run_timer, " in ", pred_latest_path, "!\n")
        raise ValueError


def get_LTR_predict(count_df, ranklib_path, ranker, pred_path, DV, MED, file_name='prediction.txt'):
    col_map = get_col_map(MED, DV)

    # include all the prediction in this setting
    all_fair_settings = [f for f in os.listdir(ranklib_path) if
                         ~os.path.isfile(os.path.join(ranklib_path, f)) and "." not in f]

    for pred_di in all_fair_settings:
        cols = col_map[pred_di.split("__")[-1]]
        train_cols = col_map[pred_di.split("__")[0]]

        ri_pred = get_prediction_scores(os.path.join(ranklib_path, pred_di), "R", ranker, file_name)
        pred_y_col = train_cols[-1] + "__" + cols[-1]
        count_df = count_df[count_df["UID"].astype(str).isin([x for x in ri_pred])]
        count_df.loc[:, pred_y_col] = count_df["UID"].apply(lambda x: ri_pred[str(x)])

    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    output_f = os.path.join(pred_path, "pred.csv")

    writeToCSV(output_f, count_df)
    print("--- Save LTR predict in ", output_f, " --- \n")
    return count_df