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
from src.scripts.utils import *
import random, math
import torch

def create_index(sessions):
    lens = np.fromiter(map(len, sessions), dtype=np.long)
    session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
    label_idx = map(lambda l: range(1, l), lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype=np.long)
    idx = np.column_stack((session_idx, label_idx))
    return idx


def read_sessions(filepath):
    sessions = load_file(str(filepath))
    if not isinstance(sessions, pd.DataFrame):
        new_sessions = []
        num_cols = len(sessions)
        if num_cols == 3:
            for i in range(len(sessions[0])):
                new_sessions.append((sessions[0][i], sessions[1][i], sessions[2][i]))
        else:
            new_sessions = [(x_i, y_i, None) for x_i, y_i in zip(sessions[0], sessions[1])]
    '''
    sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    '''
    return new_sessions


def read_dataset(dataset_dir):
    train_sessions = read_sessions(dataset_dir / 'train.txt')
    try:
        test_sessions = read_sessions(dataset_dir / 'test.txt')
    except:
        test_sessions = read_sessions(dataset_dir / 'valid.txt')
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())
    return train_sessions, test_sessions, num_items

class AugmentedDataset:
    def __init__(self, sessions, aug_scale = 2):
        self.sessions = sessions
        self.aug_scale = aug_scale
        # self.graphs = graphs
        # index = create_index(sessions)  # columns: sessionId, labelIndex
        self.index = np.arange(len(sessions) * aug_scale)
        np.random.shuffle(self.index)
        #self.fd = open("random_test.txt", 'w')

    def __getitem__(self, idx):
        #print(idx)
        real_index = int(self.index[idx] / self.aug_scale)
        # do_random only when aug_scale equals to 1 or after first hit
        do_random = True if self.aug_scale == 1 or self.index[idx] % self.aug_scale != 0 else False
        item_ids, y, sess_id, feature, wf = self.sessions[real_index]
        if do_random:
            half, total = math.ceil(len(item_ids)/2), len(item_ids)
            #half, total = int(len(item_ids)/2), len(item_ids)
            if half >= 1 and half != total:
                num_samples = random.randint(half, total)
                item_ids = item_ids[:num_samples]
                feature = feature[:num_samples]
        
        #self.fd.write(f"{item_ids}, {y}, {sess_id}\n")
        return item_ids, y, sess_id, feature, torch.tensor(wf)

    def __len__(self):
        return len(self.index)
