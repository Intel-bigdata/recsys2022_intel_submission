"""
 Copyright (c) 2022 https://github.com/SpaceLearner/SessionRec-pytorch(MIT LISENCE), 
 Intel made modification based on original MSGIFSR project, 
 reserve partial copyright for all modifications.

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 """
from concurrent.futures import process
import pandas as pd
import pickle as pkl
import logging
import timeit
import numpy as np
from tqdm import tqdm
import scipy.stats as ss
import datetime

class Timer:
    level = 1

    def __init__(self, name, level = 'INFO'):
        self.name = name
        self.level = 2
        if level == "DEBUG":
            self.level = 2
        if level == "INFO":
            self.level = 1
        if level == "WARN":
            self.level = 0

    def __enter__(self):
        self.start = timeit.default_timer()
        Timer.level += 1

    def __exit__(self, *a, **kw):
        Timer.level -= 1
        if self.level == 0:
            logging.warn(f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')
        if self.level == 1:
            logging.info(f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')
        if self.level == 2:
            logging.debug(f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')

def load_file(file_path):
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        data = pd.read_parquet(file_path)
    elif file_path.endswith(".pkl") or file_path.endswith(".txt"):
        with open(file_path, 'rb') as f:
            data = pkl.load(f)
    else:
        raise NotImplementedError(f"Unable to load {file_path}")    
    return data

def convert_to_sparse_table(pdf, row_idx = 0, col_idx = 1, val_idx = 2):
    assert(isinstance(pdf, pd.DataFrame))
    assert(pdf.shape[1] == 3)
    keys = pdf.keys().tolist()
    num_rows = pdf[keys[row_idx]].max() + 1
    num_cols = pdf[keys[col_idx]].max() + 1
    sparse_table = [[0] * num_cols for i in range(num_rows)]
    for _, row in pdf.iterrows():
        sparse_table[row[keys[row_idx]]][row[keys[col_idx]]] = row[keys[val_idx]]
    res = np.array(sparse_table)
    return res

def preprocess(train_session, kg_df, item_features_extra_df, train_target, recent_n_month = -1, candidate_list = None, add_features = False, save_path = None, enable_date_as_feature = False, enable_weighted_loss = False, extra_feat_key = [], exclude_feat_ids = []):
    if train_target is not None:
        train_target = train_target.rename(columns={'item_id': 'y'})
        train_target = train_target.rename(columns={'date': 'purchase_date'})
        train_target['purchase_date'] = pd.to_datetime(train_target["purchase_date"])
    train_session['date'] = pd.to_datetime(train_session["date"])

    divider = None
    if recent_n_month != -1 and train_target is not None:
        # get time divider
        max_time = train_target['purchase_date'].max()
        divider = max_time - pd.to_timedelta(31 * recent_n_month, unit='d')

        train_target = train_target[train_target['purchase_date'] > divider]
        train_session = train_session[train_session['date'] > divider]

    logging.info("Start to sort input data by session_id and date")
    with Timer("took "):
        train_session.sort_values(["session_id", "date"], inplace=True)

    if enable_date_as_feature:
        # add weekday and month to item
        train_session['dayofweek'] =  "dayofweek_" + train_session['date'].dt.dayofweek.astype(str)
        train_session['month'] = "m_" + train_session['date'].dt.month.astype(str)
    
    logging.info("Start to add elapse to start time and end time feature")
    with Timer("Took "):
        grouped = train_session.groupby('session_id').agg(start_time=('date','min'), end_time=('date','max'))
        train_session = train_session.merge(grouped, on='session_id', how='left')
        train_session['elapse_to_start'] = ((train_session['date'] - train_session['start_time']).dt.seconds/60).astype(int)
        train_session['elapse_to_end'] = ((train_session['end_time'] - train_session['date']).dt.seconds/60).astype(int)
        train_session['binned_elapse_to_start'] = pd.cut(train_session['elapse_to_start'], [-1, 0, 3, 15, 1434]).cat.codes
        train_session['binned_elapse_to_end'] = pd.cut(train_session['elapse_to_end'], [-1, 0, 3, 16, 1434]).cat.codes
    
    logging.info("Start to combine same session as one record")
    with Timer("took "):
        if enable_date_as_feature:
            processed = train_session.groupby("session_id", as_index = False).agg({'item_id':lambda x: list(x), 'date':lambda x: list(x), 'month': lambda x: list(x), 'dayofweek': lambda x: list(x)})
        else:
            processed = train_session.groupby("session_id", as_index = False).agg({'item_id':lambda x: list(x), 'binned_elapse_to_start':lambda x: list(x), 'binned_elapse_to_end':lambda x: list(x)})
        if train_target is not None:
            processed = train_target.merge(processed, how="inner", on="session_id")
    print(f"merged to target, length is {len(processed)}")

    if enable_date_as_feature:
        # add view duration
        logging.info("Start to get the viewing time of each session")
        max_num_items = 100
        with Timer("took "):
            duration_list_series = []
            for _, value in tqdm(processed["date"].items(), total = len(processed["date"])):
                #print(value)
                # instead of using exact duration, use rank here
                duration_list = [(value[i + 1] - value[i]).total_seconds() for i in range(len(value) - 1)]
                duration_list_rank = (len(duration_list) - ss.rankdata(duration_list).astype(int)).tolist() + [0]
                duration_list_series.append(np.array([f"dr_{i}" for i in duration_list_rank]))
                if len(duration_list_rank) > max_num_items:
                    max_num_items = len(duration_list_rank)
            processed["duration"] = pd.Series(duration_list_series)
        print(f"max num items in one session is {max_num_items}")

    # add feature column
    if add_features:
        logging.info("Start to add features to each item")
        with Timer("took "):
            kg_df['feature_category_id'] = kg_df['feature_category_id'].astype("string")
            kg_df['feature_value_id'] = kg_df['feature_value_id'].astype("string")
            kg_df["feature_merge"] = "f_" + kg_df['feature_category_id'] + "=" + kg_df['feature_value_id']
            codes, uniques = pd.factorize(kg_df["feature_merge"])
            if enable_date_as_feature:
                # add duration, dayofweek and month
                uniques = uniques.tolist() + [f"dayofweek_{i}" for i in range(7)] + [f"m_{i + 1}" for i in range(12)] + [f"dr_{i}" for i in range(max_num_items)]
            # categorify all features in item_features
            kg_df["feature"] = pd.Categorical(codes, categories=range(len(uniques)))
            num_unique_features = len(uniques)
            print(f"num_unique_features is {num_unique_features}")
            kg_dict = dict()
            for row in kg_df.to_dict('records'):
                if row['item_id'] not in kg_dict:
                    kg_dict[row['item_id']] = []
                kg_dict[row['item_id']].append(row['feature'])

            feature_list_series = []
            item_features_length_list = []
            for idx, item_id_list in tqdm(processed["item_id"].items(), total = len(processed["item_id"])):
                item_feature_list = []
                if enable_date_as_feature:
                    item_duration_list = processed["duration"][idx]
                    item_dayofweek_list = processed["dayofweek"][idx]
                    item_month_list = processed["month"][idx]
                    for item_id, dr, dw, m in zip(item_id_list, item_duration_list, item_dayofweek_list, item_month_list):
                        # we need to add item feature and other created features
                        item_features = kg_dict[item_id] + [uniques.index(dr), uniques.index(dw), uniques.index(m)]
                        item_feature_list.append(item_features)
                        item_features_length_list.append(len(item_features))
                else:
                    for item_id in item_id_list:
                        # we need to add item feature and other created features
                        item_features = kg_dict[item_id]
                        item_feature_list.append(item_features)
                        item_features_length_list.append(len(item_features))
                feature_list_series.append(item_feature_list)
            processed["feature"] = pd.Series(feature_list_series)
            print(f"avg item_features_len is {np.mean(item_features_length_list)}, max is {np.max(item_features_length_list)}")
    
    else:
        num_unique_features = -1

    # add weighted factor for session based on ts
    if train_target is not None and enable_weighted_loss:
        logging.info("Start to get the weighted factor for session based on ts")
        total_duration = processed['purchase_date'].max() - processed['purchase_date'].min()
        start_ts = processed['purchase_date'].min()
        with Timer("took "):
            weighted_factor_list_series = []
            for _, ts in tqdm(processed["purchase_date"].items(), total=len(processed["purchase_date"])):
                weighted_factor_list_series.append((ts - start_ts) / (2 * total_duration) + 0.5)
            processed["wf"] = pd.Series(weighted_factor_list_series)

    if not add_features:
        processed["feature"] = pd.Series([None] * len(processed))

    if train_target is None or not enable_weighted_loss:
        processed["wf"] = pd.Series([0] * len(processed))

    if candidate_list and train_target is not None:
        processed = processed[processed["y"].isin(candidate_list)]

    if train_target is None:
        processed['y'] = pd.Series([None] * len(processed))
        
    if save_path and train_target is not None:
        to_write = processed[["item_id", "y", "session_id", "feature", 'binned_elapse_to_start', 'binned_elapse_to_end', "purchase_date", "wf"]]
        to_write.to_parquet(save_path, compression = None)

    # exclude some feature if configured
    if extra_feat_key is not None:
        extra_feat_key_1 = [key for key in extra_feat_key if key not in ['binned_elapse_to_end', 'binned_elapse_to_start']]
        extra_feat_key_2 = [key for key in extra_feat_key if key in ['binned_elapse_to_end', 'binned_elapse_to_start']]
        processed, num_unique_features = add_extra(processed, num_unique_features, extra_feat_key_1, item_features_extra_df, divider = divider)
        processed, num_unique_features = add_sesstime(processed, num_unique_features, extra_feat_key_2)
    processed = exclude_feat(processed, exclude_feat_ids)
    
    return processed[["item_id", "y", "session_id", "feature", "wf"]].to_numpy().tolist(), num_unique_features

def load_preprocessed(file_path, item_features_extra_df_orig, recent_n_month = -1, candidate_list = None, add_features = False, enable_date_as_feature = False, extra_feat_key = [], predict = False, exclude_feat_ids = []):
    item_features_extra_df = item_features_extra_df_orig
    pd.set_option('display.max_columns', None)
    processed = load_file(file_path)
    divider = None
    if recent_n_month != -1:
        # get time divider
        max_time = processed['purchase_date'].max()
        divider = max_time - pd.to_timedelta(31 * recent_n_month, unit='d')

        processed = processed[processed['purchase_date'] > divider]
    if candidate_list:
        processed = processed[processed["y"].isin(candidate_list)]
    if not add_features:
        print(processed.shape)
        processed["feature"] = pd.Series([None] * len(processed["feature"]))
        num_unique_features = -1
    else:
        if enable_date_as_feature:
            num_unique_features = 1023
        else:
            num_unique_features = 904
            if extra_feat_key is not None:
                extra_feat_key_1 = [key for key in extra_feat_key if key not in ['binned_elapse_to_end', 'binned_elapse_to_start']]
                extra_feat_key_2 = [key for key in extra_feat_key if key in ['binned_elapse_to_end', 'binned_elapse_to_start']]
                processed, num_unique_features = add_extra(processed, num_unique_features, extra_feat_key_1, item_features_extra_df, divider = divider)
                processed, num_unique_features = add_sesstime(processed, num_unique_features, extra_feat_key_2)
            processed = exclude_feat(processed, exclude_feat_ids)


        processed["feature"] = pd.Series(processed["feature"])

    pd.set_option('display.max_columns', None)
    print(processed)
    return processed[["item_id", "y", "session_id", "feature", "wf"]].to_numpy().tolist(), num_unique_features

def add_extra(processed, num_unique_features, extra_feat_key, item_features_extra_df_orig, divider = None):
    if extra_feat_key is None:
        return processed, num_unique_features
    print(f"Start to add {extra_feat_key} to original 904 features")
    processed.reset_index(drop=True, inplace=True)
    item_features_extra_df = item_features_extra_df_orig.copy()
    num_feats = num_unique_features
    for feat_name in extra_feat_key:
        # because minimun value for one feature can be -1, add num_feats with 1 firstly
        num_feats += 1 #905
        # [ ...... ]904 + [-1, 0, 1, 2, 3] 909
        item_features_extra_df[feat_name] = item_features_extra_df[feat_name] + num_feats
        print(f"num_feats is {num_feats}")
        num_feats = (item_features_extra_df[feat_name].max() + 1)
        print(f"after add extra, num_feats is {num_feats}")
        item_feat_dict = dict((iid, fid) for iid, fid in zip(item_features_extra_df['item_id'].to_list(), item_features_extra_df[feat_name].to_list()))
        new_feature = []
        for k, x in tqdm(zip(processed['item_id'].to_list(), processed['feature'].to_list()), total = len(processed["feature"])):
            k = format_list(k)
            x = format_list(x)
            assert(len(k) == len(x))
            new_feature.append([format_list(fl) + [item_feat_dict[iid]] for iid, fl in zip(k, x)])
        processed['feature'] = pd.Series(new_feature)
    num_unique_features = num_feats
    return processed, num_unique_features

def add_sesstime(processed, num_unique_features, extra_feat_key):
    if extra_feat_key is None:
        return processed, num_unique_features
    print(f"Start to add {extra_feat_key} to original 904 features")
    processed.reset_index(drop=True, inplace=True)
    num_feats = num_unique_features
    for feat_name in extra_feat_key:
        new_max = num_feats + 4
        # because minimun value for one feature can be -1, add num_feats with 1 firstly
        # [ ...... ]904 + [0, 1, 2, 3] 909
        print(f"num_feats is {num_feats}")
        new_feature = []
        for x, fl in tqdm(zip(processed['feature'].to_list(), processed[feat_name].to_list()), total = len(processed["feature"])):
            x = format_list(x)
            fl = [f + num_feats for f in fl]
            new_feature.append([format_list(orig_fl) + [f] for orig_fl, f in zip(x, fl)])
        processed['feature'] = pd.Series(new_feature)
        num_feats = new_max
        print(f"after add extra, num_feats is {num_feats}")
    num_unique_features = num_feats
    return processed, num_unique_features

def exclude_feat(processed, exclude_feat_ids):
    if exclude_feat_ids is None:
        return processed
    if len(exclude_feat_ids) == 0:
        return processed
    print(f"Start to exclude feature {exclude_feat_ids} in original 904 features")
    processed.reset_index(drop=True, inplace=True)
    new_feature = []
    for k in tqdm(processed['feature'].to_list(), total = len(processed["feature"])):
        k = format_list(k)
        # k is [[feat0, feat8, feat1], [feat2, feat1, feat3], ... ]
        new_fl = []
        for fl in k:
            # fl is [feat0, feat8, feat1]
            fl = format_list(fl)
            new_fl.append([f for f in fl if f not in exclude_feat_ids])
        new_feature.append(new_fl)
    processed['feature'] = pd.Series(new_feature)
    return processed

def format_list(l):
    return l.tolist() if not isinstance(l, list) else l

def get_exclude_feat_list(categorical_item_features_df, exclude_feat_id):
    if exclude_feat_id is None:
        return []
    #exclude_col_name = "binned_feat_count"
    exclude_col_name = "feature_category"
    grouped = categorical_item_features_df.groupby(exclude_col_name, as_index = False).agg({'feature': lambda x: list(x)})
    tmp_dict = dict((cnt, fl) for cnt, fl in zip(grouped[exclude_col_name].to_list(), grouped['feature'].to_list()))
    exclude_feat_ids = []
    [exclude_feat_ids.extend(tmp_dict[int(i)]) for i in exclude_feat_id]
    return exclude_feat_ids