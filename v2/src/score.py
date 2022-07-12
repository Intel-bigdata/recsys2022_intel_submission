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
from collections import defaultdict
from itertools import permutations
import argparse

def two_sum_sort_v3(nums, target):
    n = len(nums)
    rt = []
    left, right = 0, n-1
    while left < right:
        sum = nums[left] + nums[right]
        if sum == target:
            rt.append([nums[left], nums[right]])
            left += 1
            right -= 1
            while left < right and nums[left] == nums[left-1]:
                left += 1
            while left < right and nums[right] == nums[right+1]:
                right -= 1
        elif sum < target:
            left += 1
            while left < right and nums[left] == nums[left-1]:
                left += 1
        elif sum > target:
            right -= 1
            while left < right and nums[right] == nums[right+1]:
                right -= 1
    return rt


def n_sum_sort_v1(n, nums, target):
    sz = len(nums)
    if n < 2 or sz < n:
        return []

    if n == 2:
        return two_sum_sort_v3(nums, target)
    
    res = []
    i = 0
    while i < sz:
        sub_res = n_sum_sort_v1(n-1, nums[i+1:], target-nums[i])
        for item in sub_res:
            res.append([nums[i]]+item)
        i += 1
        while i < sz and nums[i] == nums[i-1]:
            i += 1

    return res


def n_sum_permutation(nums, n, target):
    nums = sorted(nums)
    n_sum_res = n_sum_sort_v1(n, nums, target)
    # print(n_sum_res)
    res = []
    for n_sum in n_sum_res:
        res.extend(list(set(permutations(n_sum, n))))

    res = sorted(res, key=lambda x: x[0])

    return res


def get_weight_list(n=2, x=1):
    part = int(10 ** x)
    nums = [i/part for i in range(part+1)]
    nums = nums * n
    # print(nums)
    res = n_sum_permutation(nums, n, 1)
    print(f'combination methods: {len(res)}')
    return res


def cal_index(df, cutoff=100):
    purchase_path = args.label_file
    assert purchase_path != None

    purchases = pd.read_csv(purchase_path)

    total = df['session_id'].unique().shape[0]
    print(f'total sessions: {total}')
        
    df = df[df['rank'] <= cutoff]

    rt = pd.merge(
        df, purchases,
        on=['session_id', 'item_id'],
        how='left'
    )

    rt = rt[rt['date'].notnull()]

    rt['score'] = 1 / rt['rank']
    
    hit = rt.shape[0] / total
    mrr = rt['score'].sum() / total

    print(f"hit@{cutoff}: {hit:.5f}, mrr@{cutoff}: {mrr:.5f}")
    return mrr, hit


def get_sigle_submission_score(pred_path, cutoff=100):
    df = pd.read_csv(pred_path)
    cal_index(df, cutoff)


def gen_analysis_data(pred_path, save_path, cutoff=10):
    def encode_item(df):
        item_encode_path = 'datasets/dressipi/item.csv'
        oid2nid = pd.read_csv(item_encode_path)\
            .set_index(['item_id'])['index'].to_dict()
        item_id_new = df.item_id.map(oid2nid)
        df = df.assign(item_id=item_id_new)
        return df
    
    purchase_path = args.label_file
    assert purchase_path != None
    purchases = pd.read_csv(purchase_path)

    pred = pd.read_csv(pred_path)
    pred = pred[pred['rank'] <= cutoff]

    df = pd.merge(
        pred, purchases,
        on=['session_id', 'item_id'],
        how='left'
    )
    df['label'] = 0
    df.loc[df['date'].notnull(), 'label'] = 1

    session_label = df.groupby('session_id')['label'].sum()
    hit_sessions = session_label[session_label >= 1].index
    df_hit = df[df.session_id.isin(hit_sessions)]

    df_hit = df_hit[['session_id', 'item_id', 'label']]
    df_hit = encode_item(df_hit)
    df_hit.rename(columns = {'item_id':'pred'}, inplace = True)

    df_session = pd.read_csv(args.session_file)
    df_session = df_session.sort_values(['session_id', 'date'])
    df_session = encode_item(df_session)
    seq_id = df_session.groupby('session_id').apply(lambda x: ','.join(map(str, x.item_id)))\
        .reset_index(name='item_id')

    print(f'saving data to {save_path}...')
    df_save = pd.merge(
        seq_id, df_hit,
        how='inner', on='session_id'
    )

    df_save[['item_id', 'pred', 'label', 'session_id']]\
        .to_csv(save_path, sep='\t', header=False, index=False)

    return df_save


def get_combined_df(df_list, weights, cutoff=100):
    scored_df_list = []

    for df, weight in zip(df_list, weights):
        df['score'] = (101 - df['rank']) * weight
        scored_df_list.append(df)
    
    df3 = pd.concat(scored_df_list)

    df4 = df3.groupby(['session_id', 'item_id'])['score'].sum().reset_index(name='score')\
        .reset_index(drop=True)\
        .sort_values(by=['session_id', 'score'], ascending=[True, False])

    df4['rank'] = df4.groupby('session_id').cumcount()
    df4['rank'] = df4['rank'] + 1

    df5 = df4[df4['rank'] <= cutoff]

    return df5


def get_multi_submission_score(pred_path_list):
            
    param_dict = defaultdict(list)

    weights_list = get_weight_list(n=len(pred_path_list))

    for weights in weights_list:
        print(f'w = {weights}')

        df_list = map(lambda pred_path: pd.read_csv(pred_path), pred_path_list)

        df = get_combined_df(df_list, weights, cutoff=100)
        
        mrr, hit = cal_index(df)
        param_dict[weights] += [mrr, hit]
        del df

    best_w = sorted(param_dict.items(), key=lambda x: [-x[1][0], -x[1][1]])[0][0]

    return best_w


def save_combine_df(pred_path_list, save_path, weights):

    assert save_path != None

    print(f'saving combined df to {save_path}...')

    df_list = map(lambda pred_path: pd.read_csv(pred_path), pred_path_list)

    assert len(pred_path_list) == len(weights)

    df = get_combined_df(df_list, weights, cutoff=100)
    df = df.drop(columns=['score'])

    df[['session_id', 'item_id', 'rank']] = df[['session_id', 'item_id', 'rank']].astype(int)

    df.sort_values(["session_id", "rank"])\
        .to_csv(save_path, index=False)


def compare_two_pred(path_1, path_2, topk):
    print(f'compare ===> {path_1} <===> {path_2} ...')
    df1 = pd.read_csv(path_1)
    df2 = pd.read_csv(path_2)
    total1 = set(df1['session_id'].unique())
    total2 = set(df1['session_id'].unique())
    assert total1 == total2, 'must contain same session_id'

    assert df1['rank'].sum() == len(total1) * sum(range(1,101)), f'{path_1} wrong with rank'
    assert df1['rank'].sum() == df2['rank'].sum(), f'{path_2} wrong with rank'

    max_topk = 5 # wind-down the size for performance
    df1 = df1[df1['rank'] <= max_topk]
    df2 = df2[df2['rank'] <= max_topk]
    df = pd.merge(df1, df2, how='inner', on=['session_id', 'item_id'])

    common = df[(df['rank_x']<=topk) & (df['rank_y']<=topk)]['session_id'].nunique()

    print(f'\t top{topk} similarity score: {common / len(total1) * 100 :.2f}% = {common} / {len(total1)}')
    return 


def compare_multi_pred(path_list, topk=3):
    n = len(path_list)
    for i in range(n-1):
        for j in range(i+1, n):
            compare_two_pred(path_list[i], path_list[j], topk)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--task',
        choices=['sigle_submission_score', 'multi_submission_score', 
            'save_combine_df', 'gen_analysis_data', 'cmp_submission'],
        default='sigle_submission_score',
        help='ensemble task',
    )

    parser.add_argument(
        '--label-file', default='datasets/dressipi/valid_purchases_new.csv', 
        help='the purchase file of valid dataset'
    )

    parser.add_argument(
        '--session-file', default='datas/valid_sessions_new.csv', 
        help='the click session file of valid dataset'
    )

    parser.add_argument('--topk', type=int, default=3, help='eval on topk similarity')

    parser.add_argument('--path-list', nargs='+', help='pred path list', required=True)

    parser.add_argument('--weight-list', nargs='+', help='pred weight list')

    parser.add_argument(
        '--save-path', default='~/Downloads/submissions.csv', 
        help='the save path of the submission file'
    )

    args = parser.parse_args()
    print(args)

    if args.task == 'sigle_submission_score':
        pred_path = args.path_list[0]
        get_sigle_submission_score(pred_path)

        # python src/score.py \
        #     --task sigle_submission_score \
        #     --label-file 'datasets/dressipi/valid_purchases_new.csv' \
        #     --path-list '~/Downloads/submissions.csv'

    elif args.task == 'multi_submission_score':
        pred_path_list = args.path_list
        best_w = get_multi_submission_score(pred_path_list)
        print(f'best param w is: {best_w}')
        
        # python src/score.py \
        #     --task multi_submission_score \
        #     --label-file 'datasets/dressipi/valid_purchases_new.csv' \
        #     --path-list 'datas/pred/valid_prediction.csv' \
        #     'datas/valid_finetune_prediction.csv'

    elif args.task == 'save_combine_df':

        best_w = list(map(float, args.weight_list))
        save_path = args.save_path
        pred_path_list = args.path_list

        save_combine_df(pred_path_list, save_path, weights=best_w)

        # python src/score.py \
        #     --task save_combine_df \
        #     --label-file 'datasets/dressipi/valid_purchases_new.csv' \
        #     --path-list 'datas/pred/valid_prediction.csv' \
        #     'datas/pred/valid_finetune_prediction.csv' \
        #     --weight-list 0 1 \
        #     --save-path '~/Downloads/submissions.csv'

    elif args.task == 'cmp_submission':
        pred_path_list = args.path_list
        compare_multi_pred(pred_path_list, topk=args.topk)

        # python src/score.py \
        #     --task cmp_submission \
        #     --topk 3 \
        #     --path-list 'datas/pred/valid_prediction.csv' \
        #     'datas/pred/valid_finetune_prediction.csv' 

    elif args.task == 'gen_analysis_data':
        pred_path = args.path_list[0]
        gen_analysis_data(pred_path, args.save_path)

        # python src/score.py \
        #     --task gen_analysis_data \
        #     --label-file 'datasets/dressipi/valid_purchases_new.csv' \
        #     --session-file 'datas/valid_sessions_new.csv' \
        #     --path-list 'datas/pred/valid_prediction.csv' \
        #     --save-path '~/Downloads/submissions.csv'
