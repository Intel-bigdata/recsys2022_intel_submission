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
from multiprocessing import Pool
import numpy as np
import gc

# 有序数组，找到所有，不重复的2 个数字组合
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


# 有序数组，找到所有，不重复的 n 个数和为 target
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


# 无序数组，找到所有，不重复的 n 个数和为 target，并进行去重排列
def n_sum_permutation(nums, n, target):
    nums = sorted(nums)
    n_sum_res = n_sum_sort_v1(n, nums, target)
    # print(n_sum_res)
    res = []
    for n_sum in n_sum_res:
        res.extend(list(set(permutations(n_sum, n))))

    res = sorted(res, key=lambda x: x[0])

    return res


# 找到 n 个数, 位于[0, 1], 和为 1
# 精确到小数点后 x 位
# x = 1, 则精确到 0.1, x = 2, 精确到 0.01 位, 依次类推
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
    # print(f'total sessions: {total}')
        
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

    return mrr, hit


def get_sigle_submission_score(pred_path, cutoff=100):
    df = pd.read_csv(pred_path)
    mrr, hit = cal_index(df, cutoff)
    print(f"hit@{cutoff}: {hit:.5f}, mrr@{cutoff}: {mrr:.5f}")
    print(f"{mrr:.5f} {hit:.5f}")


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

def get_mrr_perweight(weights):
    df = get_combined_df(df_list, weights, cutoff=100)
    mrr, hit = cal_index(df)
    print(f'w = {weights}, mrr: {mrr}, hit: {hit}')
    del df
    return list(weights)+[mrr, hit]

def save_combine_df(pred_path_list, save_path, weights):
    # pred_1_path, pred_2_path 分别为较差和较好的那一个

    assert save_path != None

    print(f'saving combined df to {save_path}...')

    df_list = map(lambda pred_path: pd.read_csv(pred_path), pred_path_list)

    assert len(pred_path_list) == len(weights)

    df = get_combined_df(df_list, weights, cutoff=100)
    df = df.drop(columns=['score'])

    df[['session_id', 'item_id', 'rank']] = df[['session_id', 'item_id', 'rank']].astype(int)

    df.sort_values(["session_id", "rank"])\
        .to_csv(save_path, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t','--task',choices=['sigle_submission_score', 'multi_submission_score', 'save_combine_df'],default='sigle_submission_score')
    parser.add_argument('-p','--path-list', nargs='+', help='pred path list', required=True)
    parser.add_argument('-w','--weight-list', nargs='+', help='pred weight list')
    parser.add_argument('-s','--save-path', default='~/Downloads/submissions.csv', help='the save path of the submission file')
    parser.add_argument('--threads', default=48, type=int,help='multi processing threads')
    parser.add_argument('--label-file', default='valid_purchases_new.csv', help='the purchase file of valid dataset')

    args = parser.parse_args()
    print(args)

    if args.task == 'sigle_submission_score':
        pred_path = args.path_list[0]
        get_sigle_submission_score(pred_path)

        # python src/score.py \
        #     --task sigle_submission_score \
        #     --path-list '~/Downloads/submissions.csv'

    elif args.task == 'multi_submission_score':
        pred_path_list = args.path_list
        if not isinstance(args.weight_list, type(None)) and len(args.weight_list) == len(pred_path_list):
            df_list = map(lambda pred_path: pd.read_csv(pred_path), pred_path_list)
            weights = list(map(float, args.weight_list))
            df = get_combined_df(df_list, weights, cutoff=100)
            mrr, hit = cal_index(df)
            print(f"hit@{100}: {hit:.5f}, mrr@{100}: {mrr:.5f}")
            print(f"{mrr:.5f} {hit:.5f}")
        else:
            weights_list = get_weight_list(n=len(pred_path_list))
            df_list = [pd.read_csv(pred_path) for pred_path in pred_path_list]

            pool = Pool(args.threads)
            results = pool.map(get_mrr_perweight,weights_list)
            pool.close()
            pool.join()
            gc.collect()
            param_np = np.array(results)
            param_np = param_np[np.argsort(param_np[:,-2])]
            print(f'best param w and score is: {param_np[-1]}')
        
        # python src/score.py \
        #     --task multi_submission_score \
        #     --path-list 'datas/chendi/pred/valid_prediction.csv' \
        #     'datas/chendi/pred/valid_finetune_prediction.csv'

    elif args.task == 'save_combine_df':

        best_w = list(map(float, args.weight_list))
        save_path = args.save_path
        pred_path_list = args.path_list

        save_combine_df(pred_path_list, save_path, weights=best_w)

        # python src/score.py \
        #     --task save_combine_df \
        #     --path-list 'datas/chendi/pred/valid_prediction.csv' \
        #     'datas/chendi/pred/valid_finetune_prediction.csv' \
        #     --weight-list 0 1 \
        #     --save-path '~/Downloads/submissions.csv'

    pred_path = (
        # local valid
        '~/Downloads/submissions.csv'
        'datas/chendi/pred/sumission_feat1.csv'
        'datas/chendi/pred/sumission_feat3.csv'
        'datas/chendi/pred/valid_prediction.csv'
        'datas/chendi/pred/valid_finetune_prediction.csv'
        # colab valid
        '/content/drive/MyDrive/models/dressipi_v4-1/submissions_2.csv'
        '/content/drive/MyDrive/share/valid_prediction.csv'
        '/content/drive/MyDrive/share/valid_finetune_prediction.csv'
        # colab test
        '/content/drive/MyDrive/share/prediction_nofinetune.csv'
        '/content/drive/MyDrive/models/dressipi_v5/submissions_1940.csv'
        '/content/drive/MyDrive/share/prediction_19887.csv'
    )

    save_path = (
        '~/Downloads/submissions.csv'
        '/content/drive/MyDrive/share/submissions.csv'
    )
