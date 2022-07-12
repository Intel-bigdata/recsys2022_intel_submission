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
from pathlib import Path
import argparse
import sys
from utils.data.preprocess import (
    gen_train_file, gen_valid_file, gen_test_file,
    save_item_extend_data, gen_normal_feature_file, gen_time_feature_file
)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument(
    '-d',
    '--dataset',
    default='dressipi',
    help='the dataset name',
)
required.add_argument(
    '-f',
    '--filepath',
    help='train_sessions.csv,train_purchases.csv,item_features.csv'
)
optional.add_argument(
    '-t',
    '--dataset-dir',
    default='datasets/{dataset}',
    help='the folder to save the preprocessed dataset',
)
parser._action_groups.append(optional)
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir.format(dataset=args.dataset))


# produce 'train_12-17_month.txt', 'train_final_small_23.txt', 'train_final_small_25.txt'
gen_train_file(
    dataset_dir,
    'datas/item_features.csv',
    'datas/train_sessions.csv',
    'datas/train_purchases.csv',
    task='leaderboard'
)


# produce 'test.txt'
gen_test_file(
    dataset_dir,
    'datas/test_leaderboard_sessions.csv',
    'test.txt'
)


# produce 'test_final.txt'
gen_test_file(
    dataset_dir,
    'datas/test_final_sessions.csv',
    'test_final.txt'
)


# produce 'item_release_date.pickle'
# only contain train session data
gen_time_feature_file(
    'datas/item_extend.csv',
    '2021-06-01',
    'datasets/dressipi/item_release_date.pickle'
)

