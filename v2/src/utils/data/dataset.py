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
import itertools
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict


def create_index(sessions, mode='keep_next'):
    if mode == 'keep_next':
        lens = np.fromiter(map(len, sessions), dtype=np.int32)
        session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
        label_idx = map(lambda l: range(1, l), lens)
        label_idx = itertools.chain.from_iterable(label_idx)
        label_idx = np.fromiter(label_idx, dtype=np.int32)
        idx = np.column_stack((session_idx, label_idx))
    elif mode == 'keep_last':
        lens = np.fromiter(map(len, sessions), dtype=np.int32)
        session_idx = np.arange(len(sessions))
        label_idx = np.fromiter(lens-1, dtype=np.int32)
        idx = np.column_stack((session_idx, label_idx))
    elif mode == 'front_last':
        lens = np.fromiter(map(len, sessions), dtype=np.int32)
        session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
        seq_idx = map(lambda l: range(1, l), lens)
        seq_idx = itertools.chain.from_iterable(seq_idx)
        seq_idx = np.fromiter(seq_idx, dtype=np.int32)
        label_idx = np.repeat(lens - 1, lens - 1)
        idx = np.column_stack((session_idx, seq_idx, label_idx))
    else:
        raise NotImplementedError(f"Not supported mode: {mode}")
    return idx


def read_sessions(filepaths):
    rt = []
    for filepath in filepaths:
        sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
        sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values.tolist()
        rt += sessions
    return rt


def read_num_items(dataset_dir):
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())
    return num_items


def read_item_feat(dataset_dir):
    item_feat_pickle_path = dataset_dir / 'item_features.pickle'
    with open(item_feat_pickle_path, 'rb') as file:
        item_feat_map = pickle.load(file)

    return item_feat_map


def read_item_date(item_date_path):
    with open(item_date_path, 'rb') as file:
        item_date_map = pickle.load(file)

    return item_date_map


class AugmentedDataset:
    def __init__(self, sessions, dataset_dir, item_date_map,
        mode='keep_next', sort_by_length=False, 
        num_items=23691, num_cats=74, num_feats=895):
        self.sessions = sessions
        # self.graphs = graphs
        self.mode = mode
        index = create_index(sessions, mode)  # columns: sessionId, labelIndex

        if sort_by_length:
            # sort by labelIndex in descending order
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        self.index = index

        self.item_feat_map = read_item_feat(dataset_dir)
        self.num_items = num_items
        self.num_feats = num_feats
        self.num_cats = num_cats
        self.item_date_map = item_date_map

    def __getitem__(self, idx):
        if self.mode in ['keep_next', 'keep_last']:
            sid, lidx = self.index[idx]
            seq = self.sessions[sid][:lidx]
            label = self.sessions[sid][lidx]
        elif self.mode in ['front_last']:
            sid, seq_id, lidx = self.index[idx]
            seq = self.sessions[sid][:seq_id]
            label = self.sessions[sid][lidx]
        else:
            raise NotImplementedError(f"Not supported mode: {self.mode}")

        feat_items_dict = defaultdict(set)
        cat_items_dict = defaultdict(list)
        item_cats_dict = defaultdict(set)
        cat_feats_dict = defaultdict(list)
        feat_freq_dict = defaultdict(lambda : [0, 0])
        release_time_list = []
        for index, item in enumerate(seq):
            release_time_list.append(self.item_date_map.get(item, [0, 13, 45, 53]))
            for cat, feat in self.item_feat_map[item]:
                feat += self.num_items
                cat += self.num_items + self.num_feats
                item_cats_dict[item].add(cat)
                cat_feats_dict[cat].append(feat)
                feat_items_dict[feat].add(item)
                cat_items_dict[cat].append(item)
                feat_freq_dict[feat][0] += 1
                feat_freq_dict[feat][1] = index
        item_cats_dict = {k: ','.join(map(str, sorted(v))) for k, v in item_cats_dict.items()}

        most_freq_feat = sorted(feat_freq_dict.items(), key=lambda x: (-x[1][0], -x[1][1]))[0][0]
        
        # print(seq, label, feat_items_dict, cat_feats_dict, most_freq_feat, item_cats_dict, release_time_list)
        return seq, label, feat_items_dict, cat_feats_dict, most_freq_feat, item_cats_dict, cat_items_dict, release_time_list

    def __len__(self):
        return len(self.index)


if __name__ == '__main__':
    sessions = [
        [888,888,328,328],
        [13640,13640,10220,10220,100]
    ]

    print(create_index(sessions, mode='front_last'))
    print(create_index(sessions, mode='keep_last'))

