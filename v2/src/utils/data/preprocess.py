"""
 Copyright (c) 2022 Intel.

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
import pandas as pd
import numpy as np
import random
import pickle



def filter_short_sessions(df, min_len=2):
    session_len = df.groupby('sessionId', sort=False).size()
    long_sessions = session_len[session_len >= min_len].index
    df_long = df[df.sessionId.isin(long_sessions)]
    return df_long


def filter_no_candidate_sessions(df):
    # print(df.shape)
    candidate_items_path = 'datasets/dressipi/valid_candidate_items.csv'
    candidate_data = pd.read_csv(candidate_items_path)
    candidate_data['is_candidate'] = 1
    candidate_data = pd.merge(
        df.sort_values(['sessionId', 'timestamp']).groupby('sessionId').tail(1), 
        candidate_data, 
        how='left', 
        left_on='itemId',
        right_on='item_id'
    ).fillna(0)
    candidate_data = candidate_data.groupby('sessionId')['is_candidate'].sum()
    df = df[df.sessionId.isin(candidate_data[candidate_data > 0].index)]
    # print(df.shape)
    # print(df.groupby('sessionId').size().shape)
    return df


def filter_infreq_items(df, min_support=5):
    item_support = df.groupby('itemId', sort=False).size()
    freq_items = item_support[item_support >= min_support].index
    df_freq = df[df.itemId.isin(freq_items)]
    return df_freq


def filter_until_all_long_and_freq(df, min_len=2, min_support=5):
    while True:
        df_long = filter_short_sessions(df, min_len)
        df_freq = filter_infreq_items(df_long, min_support)
        if len(df_freq) == len(df):
            break
        df = df_freq
    return df


def truncate_long_sessions(df, max_len=20, is_sorted=False):
    if not is_sorted:
        df = df.sort_values(['sessionId', 'timestamp'])
    itemIdx = df.groupby('sessionId').cumcount()
    df_t = df[itemIdx < max_len]
    return df_t


def filter_target_month(df, month_list):
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['month_past'] = (df['year'] - 2020)*12 + df['month']
    df_month = df[
        df['month_past'].isin(month_list)
    ]
    return df_month


def filter_short_diff(df, min_diff=1):
    df = df[(df['next_time_diff'] == -1) | (df['next_time_diff'] >= min_diff)]
    return df


def update_id(df, field):
    labels = pd.factorize(df[field])[0]
    kwargs = {field: labels}
    df = df.assign(**kwargs)
    return df


def remove_immediate_repeats(df):
    df_prev = df.shift()
    is_not_repeat = (df.sessionId != df_prev.sessionId) | (df.itemId != df_prev.itemId)
    df_no_repeat = df[is_not_repeat]
    return df_no_repeat


def reorder_sessions_by_endtime(df):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    df_endtime = endtime.sort_values().reset_index()
    oid2nid = dict(zip(df_endtime.sessionId, df_endtime.index))
    sessionId_new = df.sessionId.map(oid2nid)
    df = df.assign(sessionId=sessionId_new)
    df = df.sort_values(['sessionId', 'timestamp'])
    return df


def keep_top_n_items(df, n):
    item_support = df.groupby('itemId', sort=False).size()
    top_items = item_support.nlargest(n).index
    df_top = df[df.itemId.isin(top_items)]
    return df_top


def split_by_time(df, timedelta):
    max_time = df.timestamp.max()
    end_time = df.groupby('sessionId').timestamp.max()
    split_time = max_time - timedelta
    train_sids = end_time[end_time < split_time].index
    test_sids = end_time[end_time > split_time].index
    df_train = df[df.sessionId.isin(train_sids)]
    df_test = df[df.sessionId.isin(test_sids)]
    return df_train, df_test


def train_test_split(df, test_split=0.2):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    endtime = endtime.sort_values()
    num_tests = int(len(endtime) * test_split)
    test_session_ids = endtime.index[-num_tests:]
    df_train = df[~df.sessionId.isin(test_session_ids)]
    df_test = df[df.sessionId.isin(test_session_ids)]
    return df_train, df_test


def save_sessions(df, filepath):
    df = reorder_sessions_by_endtime(df)
    seq_id = df.groupby('sessionId').apply(lambda x: ','.join(map(str, x.itemId)))\
        .reset_index(name='itemId')
    seq_time = df.groupby('sessionId').apply(lambda x: ','.join(map(str, x.next_time_diff)))\
        .reset_index(name='next_time_diff')
    sessions = pd.merge(
        seq_id, seq_time,
        how='left', on='sessionId'
    )
    # keys = ['itemId', 'next_time_diff']
    keys = ['itemId']
    sessions[keys]\
        .to_csv(filepath, sep='\t', header=False, index=False)


def save_test_sessions(df, filepath):
    sessions = df.groupby('sessionId').apply(lambda x: ','.join(map(str, x.itemId)))\
        .reset_index(name='itemId')\
        .apply(lambda x: f'{x.itemId},{x.sessionId}', axis=1)
    sessions.to_csv(filepath, sep='\t', header=False, index=False)


def preprocess_item(dataset_dir, item_path):
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f'reading {item_path}...')
    # item_id,feature_category_id,feature_value_id
    df = pd.read_csv(
        item_path,
        usecols=[0, 1, 2],
        delimiter=',',
    )
    df.rename(columns={'item_id': 'itemId'}, inplace = True)

    # encode itemId 
    print(f'No. of Items: {df.itemId.nunique()}')
    itemId_new, uniques = pd.factorize(df.itemId)
    pd.DataFrame({'item_id':uniques}).reset_index()\
        .to_csv(f'{dataset_dir}/item.csv', index=False)
    num_items = len(uniques)
    with open(dataset_dir / 'num_items.txt', 'w') as f:
        f.write(str(num_items))


def encode_item(dataset_dir, df):
    # update itemId
    oid2nid = pd.read_csv(f'{dataset_dir}/item.csv')\
        .set_index(['item_id'])['index'].to_dict()
    itemId_new = df.itemId.map(oid2nid)
    df = df.assign(itemId=itemId_new)
    return df


def preprocess_train(dataset_dir, df_train, file_name, filter_process=False):
    if filter_process:
        # filter session that does not contain candidate_item 的 session
        # df_train = filter_no_candidate_sessions(df_train)

        # filter sessions whose length larger than x
        # df_train = truncate_long_sessions(df_train, max_len=40, is_sorted=False)

        # filter infrequent item
        # df_train = filter_infreq_items(df_train, min_support=5)

        # filter sessions whose length less than 2
        # df_train = filter_short_sessions(df_train, min_len=2)
        pass
    
    df_train = encode_item(dataset_dir, df_train)

    save_path = dataset_dir / file_name
    print(f'saving dataset to {save_path}')
    save_sessions(df_train, save_path)

    return df_train


def preprocess_valid(dataset_dir, df_valid, file_name='valid.txt'):
    df_valid = encode_item(dataset_dir, df_valid)

    save_path = dataset_dir / file_name
    print(f'saving dataset to {save_path}')
    save_sessions(df_valid, save_path)
    return df_valid


def preprocess_test(dataset_dir, df_test, file_name='test.txt'):
    df_test = encode_item(dataset_dir, df_test)

    save_path = dataset_dir / file_name
    print(f'saving dataset to {save_path}')
    save_test_sessions(df_test, save_path)

    return df_test


def get_session_df(path):
    print(f'reading {path}...')
    df = pd.read_csv(
        path,
        usecols=[0, 1, 2],
        delimiter=',',
        parse_dates=['date'],
        infer_datetime_format=True,
    )
    df.rename(
        columns={
            'date':'timestamp', 'session_id':'sessionId',
            'item_id': 'itemId'
        }, 
        inplace = True
    )
    df.sort_values(['sessionId', 'timestamp'], inplace=True)

    df['next_time'] = df.groupby('sessionId')['timestamp'].shift(-1)
    df['next_time_diff'] = df.apply(
        lambda x: (x['next_time'] - x['timestamp']).total_seconds(), 
        axis=1
    )
    df.fillna(-1, inplace=True)
    df.drop(columns=['next_time'], inplace=True)
    return df


def filter_random_length(df):
    def f(n):
        low, high = 4, 75
        if n <= low:
            return n
        if n >= high:
            n = high
        n = int(round(0.5 * (random.random() + 1) * n, 0))
        return n

    random.seed(2022)
    session_len = df.groupby('sessionId').size().apply(f).reset_index(name='length')
    df = pd.merge(df, session_len, how='left', on='sessionId')
    df = df.sort_values(['sessionId', 'timestamp'])
    df['index'] = df.groupby('sessionId').cumcount()
    df_t = df[df['index'] < df['length']]
    print(f'After random split: {df_t.shape}')
    return df_t[['sessionId', 'itemId', 'timestamp']]


def preprocess_data(df):
    # you can put your preprocess here, below is some example

    # filter infrequent item
    # df = filter_infreq_items(df, min_support=5)

    # filter long sessions
    # df = truncate_long_sessions(df, max_len=40, is_sorted=False)

    return df


def gen_train_file(dataset_dir, item_path, session_path, purchase_path, task='leaderboard'):

    print('start preprocessing')

    df = get_session_df(session_path)
    df = preprocess_data(df)

    if item_path:
        preprocess_item(dataset_dir, item_path)

    # concat click data with purchase data
    df_purchase = get_session_df(purchase_path)
    df = pd.concat([df, df_purchase]).reset_index(drop=True)
    
    # filter out target month
    if task == 'leaderboard':
        df = filter_target_month(df, list(range(12,18)))
        preprocess_train(dataset_dir, df, 'train_12-17_month.txt', True)
    elif task == 'local_valid':
        df = filter_target_month(df, list(range(11,17)))
        preprocess_train(dataset_dir, df, 'train_11-16_month.txt', True)

    # save last 23day data
    df_train, df_valid = split_by_time(df, pd.Timedelta(days=23))
    if task == 'leaderboard':
        df_valid = preprocess_valid(dataset_dir, df_valid, 'train_final_small_23.txt')
    elif task == 'local_valid':
        df_valid = preprocess_valid(dataset_dir, df_valid, 'train_small_23.txt')

    # save last 25day data
    df_train, df_valid = split_by_time(df, pd.Timedelta(days=25))
    if task == 'leaderboard':
        df_valid = preprocess_valid(dataset_dir, df_valid, 'train_final_small_25.txt')
    elif task == 'local_valid':
        df_valid = preprocess_valid(dataset_dir, df_valid, 'train_small_25.txt')
    
    print(f'No. of Clicks: {len(df_train) + len(df_valid)}')


def gen_valid_file(dataset_dir, session_path, save_path):
    print('start preprocessing')

    df = get_session_df(session_path)
    df = preprocess_data(df)

    df_test = preprocess_valid(dataset_dir, df, save_path)
    print(f'No. of Clicks: {len(df_test)}')


def gen_test_file(dataset_dir, session_path, save_path):
    print('start preprocessing')

    df = get_session_df(session_path)
    df = preprocess_data(df)

    df_test = preprocess_test(dataset_dir, df, save_path)
    print(f'No. of Clicks: {len(df_test)}')


def get_session_extend_data(sessions_path, sessions_extend_save_path, 
                            candidate_items_path,
                            purchases_path=''):
    train_sessions_data = pd.read_csv(sessions_path)
    train_sessions_data.sort_values(by=['session_id', 'date'], inplace = True)
    
    train_sessions_data['view_item_rank'] = train_sessions_data\
        .groupby(['session_id'])['session_id'].rank(method='first')
    train_sessions_data['view_size'] = train_sessions_data.groupby(['session_id'])['item_id'].transform('size')
    
    train_sessions_data_1 = train_sessions_data[['session_id', 'view_item_rank', 'date']]\
        .rename(columns={'date':'pre_item_view_date'})
    train_sessions_data_1['view_item_rank'] = train_sessions_data_1['view_item_rank'] + 1
    
    train_sessions_data_2 = train_sessions_data[['session_id', 'view_item_rank', 'date']]\
        .rename(columns={'date':'next_item_view_date'})
    train_sessions_data_2['view_item_rank'] = train_sessions_data_2['view_item_rank'] - 1
    
    temp_1 = pd.merge(train_sessions_data, train_sessions_data_1, how='left', on=['session_id', 'view_item_rank'])
    del train_sessions_data, train_sessions_data_1
    
    temp_1 = pd.merge(temp_1, train_sessions_data_2, how='left', on=['session_id', 'view_item_rank'])
    del train_sessions_data_2
    
    temp_1 = temp_1.fillna('')
    
    for date_name in ['date', 'pre_item_view_date', 'next_item_view_date']:
        temp_1[date_name] = pd.to_datetime(temp_1[date_name])
    
    if purchases_path:
        train_purchases_data = pd.read_csv(purchases_path)
        train_purchases_data.rename(columns={'date':'purchase_date', 'item_id':'purchase_item_id'}, inplace = True)
        temp_1 = pd.merge(temp_1, train_purchases_data, how='left', on=['session_id'])
        del train_purchases_data
        
        for date_name in ['purchase_date']:
            temp_1[date_name] = pd.to_datetime(temp_1[date_name])
    
        temp_1["next_item_view_date"].fillna(temp_1.purchase_date, inplace=True)
    
    temp_1['next_view_time_diff'] = temp_1.apply(
        lambda x: (x['next_item_view_date'] - x['date']).total_seconds(), axis=1)

    temp_1['pre_view_time_diff'] = temp_1.apply(
        lambda x: (x['date'] - x['pre_item_view_date']).total_seconds(), axis=1)
    
    candidate_data_1 = pd.read_csv(candidate_items_path)
    candidate_data_1['is_view_candidate'] = 1
    temp_2 = pd.merge(temp_1, candidate_data_1, how='left', on=['item_id'])

    if purchases_path:
        candidate_data_2 = pd.read_csv(candidate_items_path)\
            .rename(columns={'item_id':'purchase_item_id'})
        candidate_data_2['is_purchase_candidate'] = 1
        temp_2 = pd.merge(temp_2, candidate_data_2, how='left', on=['purchase_item_id'])
        del candidate_data_2
    
    temp_3 = temp_2.fillna(0)

    del temp_1, temp_2, candidate_data_1
    
    temp_3['day_of_week'] = temp_3['date'].dt.weekday
    temp_3['month'] = temp_3['date'].dt.month
    temp_3['year'] = temp_3['date'].dt.year
    temp_3['month_past'] = (temp_3['year'] - 2020)*12 + temp_3['month']
    temp_3['view_hour'] = temp_3['date'].dt.hour
    
    if purchases_path:
        temp_3['purchase_hour'] = temp_3['purchase_date'].dt.hour
    
    temp_3.to_csv(sessions_extend_save_path, index=False)


def get_item_extend_data(train_session_extend, item_extend_path, candidate_items_path, 
                         train_start_dt='2020-01-01', train_end_dt='2021-06-01'):

    train_session_extend['end_dt'] = train_end_dt
    for date_name in ['date', 'purchase_date', 'end_dt']:
        train_session_extend[date_name] = pd.to_datetime(train_session_extend[date_name])
        
    temp_1 = pd.merge(
        train_session_extend['item_id'].value_counts().reset_index(name='view_count')\
            .rename(columns={'index':'item_id'}),
        train_session_extend[train_session_extend['view_item_rank'] == 1]['purchase_item_id']\
            .value_counts().reset_index(name='purchase_count')\
            .rename(columns={'index':'item_id'}),
        how='left',
        on='item_id'
    ).fillna(0).sort_values(by=['purchase_count', 'view_count'], ascending=False)\
        .reset_index(drop=True)
    
    candidate_data_1 = pd.read_csv(candidate_items_path)
    candidate_data_1['is_candidate'] = 1
    temp_2 = pd.merge(temp_1, candidate_data_1, how='left', on=['item_id'])
    
    temp_2.fillna(0, inplace=True)
    
    last_purchase = train_session_extend[train_session_extend['view_item_rank'] == 1]\
        [['purchase_item_id', 'purchase_date']]\
        .groupby('purchase_item_id').max().reset_index()\
        .rename(columns={'purchase_item_id':'item_id', 'purchase_date':'last_purchase_date'})

    first_purchase = train_session_extend[train_session_extend['view_item_rank'] == 1]\
        [['purchase_item_id', 'purchase_date']]\
        .groupby('purchase_item_id').min().reset_index()\
        .rename(columns={'purchase_item_id':'item_id', 'purchase_date':'first_purchase_date'})

    last_view = train_session_extend[['item_id', 'date']]\
        .groupby('item_id').max().reset_index()\
        .rename(columns={'date':'last_view_date'})

    first_view = train_session_extend[['item_id', 'date']]\
        .groupby('item_id').min().reset_index()\
        .rename(columns={'date':'first_view_date'})
    
    temp_3 = pd.merge(temp_2, first_purchase, how='left', on=['item_id'])
    temp_3 = pd.merge(temp_3, last_purchase, how='left', on=['item_id'])
    temp_3 = pd.merge(temp_3, last_view, how='left', on=['item_id'])
    temp_3 = pd.merge(temp_3, first_view, how='left', on=['item_id'])
    
    temp_3['start_dt'] = train_start_dt
    temp_3['end_dt'] = train_end_dt

    for date_name in ['start_dt', 'end_dt', 'last_purchase_date', 'first_purchase_date', 'last_view_date']:
        temp_3[date_name] = pd.to_datetime(temp_3[date_name])
        
    temp_3['first_purchase_passed_days'] = temp_3.apply(
        lambda x: (x['end_dt']-x['first_purchase_date']).days, axis=1
    )
    temp_3['last_purchase_passed_days'] = temp_3.apply(lambda x: (x['end_dt']-x['last_purchase_date']).days, axis=1)
    temp_3['last_view_passed_days'] = temp_3.apply(lambda x: (x['end_dt']-x['last_view_date']).days, axis=1)
    temp_3['first_view_passed_days'] = temp_3.apply(lambda x: (x['end_dt']-x['first_view_date']).days, axis=1)
    
    temp_3['release_date'] = temp_3.apply(
        lambda x: x['first_view_date'] if pd.isna(x['first_purchase_date']) 
            else min(x['first_view_date'], x['first_purchase_date']),
        axis=1
    )
    
    temp_3.fillna(0, inplace=True)
    
    temp_3.drop(columns=['start_dt', 'end_dt'], inplace=True)
    
    temp_3.to_csv(item_extend_path, index=False)


def save_item_extend_data(
    train_sessions_path, train_session_extend_save_path,
    candidate_items_path, train_purchases_path,
    item_extend_path
):
    get_session_extend_data(
        train_sessions_path, train_session_extend_save_path, 
        candidate_items_path,
        purchases_path=train_purchases_path
    )

    train_session_extend = pd.read_csv(train_session_extend_save_path)
    get_item_extend_data(train_session_extend, item_extend_path, candidate_items_path)


def gen_normal_feature_file(item_extend_path, item_feat_path, item_feat_pickle_path):
    def get_feat_index_from_df(df):
        id_new, uniques = pd.factorize(df['feature_value_id'], sort=True)
        df['feature_value_id'] = id_new
        df['feature_value_id'] = df['feature_value_id'] + 1
        return df


    def save_feat_pickle_from_path(item_feat_df, item_feat_pickle_path):
        df = get_feat_index_from_df(item_feat_df)
        df['feat'] = df.apply(lambda x: f"{x.feature_category_id}:{x.feature_value_id}", axis=1)

        df = df.groupby('item_id')['feat'].apply(lambda x: ';'.join(x)).reset_index(name='feats')
        df = encode_item(df)

        with open(item_feat_pickle_path, 'wb') as file:
            d = df\
                .apply(lambda x: list(map(lambda y: list(map(int, y.split(':'))), x.feats.split(';'))), axis=1)\
                .to_dict()

            pickle.dump(d, file)

        # with open(item_feat_pickle_path, 'rb') as file:
        #     d = pickle.load(file)
        
        return len(d)
        
        
    def get_view_cnt_df():
        df_origin = pd.read_csv(item_feat_path)
        print(df_origin.shape)

        item_extend = pd.read_csv(item_extend_path, parse_dates=['release_date'])
        
        # 25%：[0,14,95,263,14714]
        # 20%：[0, 7, 53, 148, 319, 14714]
        print(item_extend['view_count'].describe(percentiles=[.2, .4, .6, .8]))
        item_extend['view_cnt_cat'] = pd.cut(item_extend['view_count'], [0, 7, 53, 148, 319, 14714], labels=False)
        
        df = item_extend[['item_id', 'view_cnt_cat']].copy()
        df['feature_category_id'] = 74
        df['feature_value_id'] = df['view_cnt_cat'] + 1000
        df = df[['item_id', 'feature_category_id', 'feature_value_id']].copy()
        
        df = pd.concat([df_origin, df])
        print(df.shape)
        return df


    df = get_view_cnt_df()
    save_feat_pickle_from_path(df, item_feat_pickle_path)


def gen_time_feature_file(item_extend_path, start_dt, feat_save_path):
    item_extend = pd.read_csv(item_extend_path, parse_dates=['release_date'])

    item_extend['month'] = item_extend['release_date'].dt.month
    item_extend['day'] = item_extend['release_date'].dt.day + 13
    item_extend['weekday'] = item_extend['release_date'].dt.weekday + 1 + 45
    item_extend['hour'] = item_extend['release_date'].dt.hour + 1 + 53

    item_extend['release_time'] = item_extend.apply(
        lambda x: [x.month, x.day, x.weekday, x.hour],
        axis=1
    )
                
    df = encode_item(
        item_extend[item_extend['release_date'] < start_dt]
    )[['item_id', 'release_time']]

    d = df.set_index(['item_id'])['release_time'].to_dict()

    with open(feat_save_path, 'wb') as file:
        
        pickle.dump(d, file)
