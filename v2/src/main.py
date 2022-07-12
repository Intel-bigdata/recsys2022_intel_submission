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
import argparse
import os
import numpy as np
import pandas as pd
import torch
import random
import sys
sys.path.append('..')
sys.path.append('../..')

from pathlib import Path
import os
import numpy as np
import torch as th
from torch.utils.data import DataLoader, SequentialSampler
from src.utils.data.dataset import (
    read_sessions, read_num_items, 
    read_item_date, AugmentedDataset
)
from src.utils.data.collate import (
    seq_to_ccs_graph,
    collate_fn_factory_ccs
)
from src.utils.train import TrainRunner
import importlib



def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    
seed_torch(123)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    # memory_available = memory_available[1:6]
    if len(memory_available) == 0:
        return -1
    return int(np.argmax(memory_available))

os.environ["CUDA_VISIBLE_DEVICES"] = str(get_freer_gpu())
print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--task',
    choices=['train', 'eval', 'predict'],
    default='train',
    help='model task',
)
parser.add_argument(
    '--model-type', default='sihg4sr', help='the model architecture'
)
parser.add_argument(
    '--dataset-dir', default='../datasets/sample', help='the dataset directory'
)
parser.add_argument(
    '--candidate-file', default='candidate_items.csv', help='the candidate file'
)
parser.add_argument(
    '--item-date-file', default='valid_item_release_date.pickle', help='the item date file'
)
parser.add_argument(
    '--train-dataset', default='train.txt', help='the train dataset'
)
parser.add_argument(
    '--train-mode', default='keep_last', help='the train mode'
)
parser.add_argument(
    '--valid-dataset', default='valid.txt', help='the valid dataset'
)
parser.add_argument(
    '--test-dataset', default='test.txt', help='the test dataset'
)
parser.add_argument(
    '--submission-name', default='submission.csv', help='the submission name'
)
parser.add_argument(
    '--model-dir', default='../models/sample', help='the models directory'
)
parser.add_argument('--embedding-dim', type=int, default=128, help='the embedding size')
parser.add_argument('--num-layers', type=int, default=1, help='the number of layers')
parser.add_argument(
    '--feat-drop', type=float, default=0.1, help='the dropout ratio for features'
)
parser.add_argument(
    '--edge-drop', type=float, default=0.0, help='the dropout ratio for edges'
)
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
parser.add_argument(
    '--batch-size', type=int, default=512, help='the batch size for training'
)
parser.add_argument(
    '--epochs', type=int, default=30, help='the number of training epochs'
)
parser.add_argument(
    '--weight-decay',
    type=float,
    default=1e-4,
    help='the parameter for L2 regularization',
)
parser.add_argument(
    '--patience',
    type=int,
    default=1,
    help='the number of epochs that the performance does not improves after which the training stops',
)
parser.add_argument(
    '--num-workers',
    type=int,
    default=0,
    help='the number of processes to load the input graphs',
)
parser.add_argument(
    '--log-interval',
    type=int,
    default=100,
    help='print the loss after this number of iterations',
)
parser.add_argument(
    '--order',
    type=int,
    default=3,
    help='order of msg',
)
parser.add_argument(
    '--reducer',
    type=str,
    default='mean',
    help='method for reducer',
)
parser.add_argument(
    '--norm',
    type=bool,
    default=True,
    help='whether use l2 norm',
)
parser.add_argument(
    '--extra',
    action='store_false',
    help='whether use REnorm.',
)
parser.add_argument(
    '--finetune',
    action='store_true',
    help='whether finetune from a new dataset',
)
parser.add_argument(
    '--fusion',
    action='store_false',
    help='whether use IFR.',
)

args = parser.parse_args()
print(args)


device = th.device('cuda' if th.cuda.is_available() else 'cpu')
pin_memory = True if th.cuda.is_available() else False
dataset_dir = Path(args.dataset_dir)
model_dir = Path(args.model_dir)

num_items = read_num_items(dataset_dir)
item_date_map = read_item_date(dataset_dir / args.item_date_file)
collate_fn = collate_fn_factory_ccs((seq_to_ccs_graph,), order=args.order, edge_drop_ratio=args.edge_drop)


def get_train_loader():
    print('reading training dataset...')
    train_path_list = [dataset_dir / filepath for filepath in args.train_dataset.split(',')]
    train_sessions = read_sessions(train_path_list)

    train_set = AugmentedDataset(train_sessions, dataset_dir, item_date_map, 
        mode=args.train_mode, num_items=num_items)

    print(f'train_set: {len(train_set)}')

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        sampler=SequentialSampler(train_set)
    )

    return train_loader


def get_valid_loader():
    print('reading valid dataset...')
    valid_sessions = read_sessions([dataset_dir / args.valid_dataset])

    valid_set  = AugmentedDataset(valid_sessions, dataset_dir, item_date_map, 
        mode='keep_last', num_items=num_items)

    print(f'valid_set: {len(valid_set)}')

    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        # shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )

    return valid_loader


def get_candidate_nid():
    candidate_file = args.candidate_file
    oid2nid = pd.read_csv(dataset_dir / 'item.csv')\
        .set_index(['item_id'])['index'].to_dict()
    candidate_nid = sorted(
        pd.read_csv(dataset_dir / candidate_file)['item_id']\
            .map(oid2nid).values.tolist()
    )
    return candidate_nid


candidate_nid = get_candidate_nid()

SIHG4SR = importlib.import_module(f'src.models.{args.model_type}').SIHG4SR
model = SIHG4SR(
    num_items, item_date_map, args.embedding_dim, args.num_layers, 
    dropout=args.feat_drop, reducer=args.reducer, order=args.order, 
    norm=args.norm, extra=args.extra, fusion=args.fusion, device=device,
)

model = model.to(device)

runner = TrainRunner(
    model_dir,
    model,
    is_finetune = args.finetune,
    device=device,
    lr=args.lr,
    weight_decay=args.weight_decay,
    patience=args.patience,
)

if args.task == 'train':
    print(model)

    print('start training...')

    runner.train_loader = get_train_loader()
    runner.valid_loader = get_valid_loader()
    mrr, hit = runner.train(args.epochs, candidate_nid, log_interval=args.log_interval)
    print('max of all epochs: MRR@100\tHR@100')
    print(f'{mrr * 100:.3f}%\t{hit * 100:.3f}%')

elif args.task == 'eval':
    print('start evaluating...')

    cutoff = 100
    valid_loader = get_valid_loader()
    mrr, hit, loss = runner.evaluate(valid_loader, candidate_nid, cutoff=cutoff)
    print(f'MRR@{cutoff} = {mrr * 100:.3f}%, HR@{cutoff} = {hit * 100:.3f}%, Loss = {loss:.4f}')

elif args.task == 'predict':
    print('start predicting...')

    test_sessions = read_sessions([dataset_dir / args.test_dataset])
    test_set  = AugmentedDataset(test_sessions, dataset_dir, item_date_map, 
        mode='keep_last', num_items=num_items)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    candidate_oid = sorted(
        pd.read_csv(dataset_dir / args.candidate_file)['item_id']\
            .values.tolist()
    )
    candidate_index2oid = dict(zip(range(len(candidate_oid)), candidate_oid))

    df_predict = runner.predict(test_loader, candidate_nid, candidate_index2oid, cutoff=100)
    df_predict.to_csv(model_dir / args.submission_name, index=False)
else:
    pass
