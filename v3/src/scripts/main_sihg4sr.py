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
import torch
import random
import sys
import os, sys, pathlib
import logging

try:
    from importlib import reload
    reload(logging)
except:
    pass
logging.basicConfig(level=logging.INFO, stream = sys.stdout)

current_path = str(pathlib.Path(__file__).parent.absolute())
logging.info(current_path)
sys.path.append(f'{current_path}/..')
sys.path.append(f'{current_path}/../..')


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
seed_torch(123)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [
        int(x.split()[2]) for x in open('tmp', 'r').readlines()
    ]
    # memory_available = memory_available[1:6]
    if len(memory_available) == 0:
        return -1
    return int(np.argmax(memory_available))
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(get_freer_gpu())
print(f"CUDA_VISIBLE_DEVICES is {os.environ['CUDA_VISIBLE_DEVICES']}")


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--embedding-dim',type=int,default=256,help='the embedding size')
parser.add_argument('--num-layers',type=int,default=1,help='the number of layers')
parser.add_argument('--feat-drop',type=float,default=0.1,help='the dropout ratio for features')
parser.add_argument('--lr', type=float,default=1e-3, help='the learning rate')
parser.add_argument('--weight-decay',type=float,default=1e-4,help='the parameter for L2 regularization')
parser.add_argument('--patience',type=int,default=5,help='the number of epochs that the performance does\
    not improves after which the training stops',)
parser.add_argument('--num-workers',type=int,default=4,help='the number of processes to load the input graphs',)
parser.add_argument('--valid-split',type=float,default=None,help='the fraction for the validation set',)
parser.add_argument('--log-interval',type=int,default=100,help='print the loss after this number of iterations',)
parser.add_argument('--order',type=int,default=1,help='order of msg',)
parser.add_argument('--reducer',type=str,default='mean',help='method for reducer',)
parser.add_argument('--norm',type=bool,default=True,help='whether use l2 norm',)
parser.add_argument('--extra',action='store_true',help='whether use REnorm.',)
parser.add_argument('--fusion', action='store_true', help='whether use IFR.')
parser.add_argument('--topk', type=int, default=100, help='topk')
parser.add_argument('--test-with-random-sample', action='store_true', help='test dataset will be random sampled')
parser.add_argument('--data-augment', default=1, type=int, help='the scale of data augment')
parser.add_argument('--use-target-output', action='store_true', help='test')
parser.add_argument('--enable-features-gnn', action='store_true', help='enable item features to add a knowledge graph to GNN')
parser.add_argument('--item-features', default='item_features.csv', type=str)
parser.add_argument('--attent-longest-view', action='store_true', help='instead of using last item to add attend, use item with longest view duration instead')
parser.add_argument('--enable_transformer', action='store_true')
parser.add_argument('--transformer_layer', default=2, type=int)
parser.add_argument('--transformer_head', default=8, type=int)
parser.add_argument('--enable_date_as_feature', action='store_true')

parser.add_argument('--save_path', default='model_save/', type=str)
parser.add_argument('--enable_save_logits', action='store_true')
parser.add_argument('--batch-size',type=int,default=512,help='the batch size for training')
parser.add_argument('--epochs',type=int,default=15,help='the number of training epochs')
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--predict', action='store_true', help='predict')
parser.add_argument('--predict-valid', action='store_true', help='predict')
parser.add_argument('--predict-leaderboard', action='store_true', help='predict')
parser.add_argument('--predict-final', action='store_true', help='predict')
parser.add_argument('--finetune', action='store_true', help='finetune option would config resume_train, epochs, and use-recent-n-month sort-train-data at once')
parser.add_argument('--resume-train', action='store_true', help='resume train with existing model')
parser.add_argument('--model',type=str,default='model_save/0323/model_0_0.168.pth',help='saved model path',)
parser.add_argument('--dataset-dir',default=f'../../data/',help='the dataset directory')
parser.add_argument('--train-session', default='train_sessions_new.csv', type=str)
parser.add_argument('--train-purchase', default='train_purchases_new.csv', type=str)
parser.add_argument('--valid-session', default='valid_sessions_new.csv', type=str)
parser.add_argument('--valid-purchase', default='valid_purchases_new.csv', type=str)
parser.add_argument('--candidate-list', default='valid_candidate_items.csv', type=str)
parser.add_argument('--use-recent-n-month', default=-1, type=float, help='use recent months of data to train, -1 means not enable, n means recent n months')
parser.add_argument('--enable-weighted-loss', action='store_true', help='enable weighted loss based on ts')
parser.add_argument('--sort-train-data', action='store_true', help='sort training data')
parser.add_argument('--train-clickthrough', action='store_true', help='train click through')
parser.add_argument('--train-clickthrough-extra', action='store_true', help='train click through')
parser.add_argument('--srl_ratio',type=int,default=0.7)
parser.add_argument('--srg_ratio',type=int,default=0.3)
parser.add_argument('--prepare_data', action='store_true')
parser.add_argument('--session_under_5', action='store_true')
parser.add_argument('--session_above_5', action='store_true')
parser.add_argument('--extra_feat_key', nargs='+', help='binned_publish_time, binned_count_item_clicks, binned_elapse_to_end, itemcat_fp')
parser.add_argument('--itemcat_bin_size',type=int,default=100)
parser.add_argument('--exclude_feat_id', nargs='+', help='0, 1, 2, 3, 4, 5, 6, 7, 8')

args = parser.parse_args()
#args.use_target_output = True
# === default ====
args.extra = True
args.enable_features_gnn = True
args.enable_weighted_loss = True
if args.extra_feat_key is None:
    args.extra_feat_key = []
'''
default_extra_feat_key = ['binned_count_item_clicks', 'binned_elapse_to_end']
args.extra_feat_key = default_extra_feat_key + [i for i in args.extra_feat_key if i not in default_extra_feat_key]
'''
# === finetune options ===
if args.finetune:
    args.epochs = 4
    if args.use_recent_n_month == -1:
        args.use_recent_n_month = 0.83
    args.resume_train = True
# === predict options ===
if args.predict_leaderboard:
    args.predict = True
    args.candidate_list = "candidate_items.csv"
    args.valid_session  = "test_leaderboard_sessions.csv"
    args.valid_purchase = ""
if args.predict_final:
    args.predict = True
    args.candidate_list = "candidate_items.csv"
    args.valid_session  = "test_final_sessions.csv"
    args.valid_purchase = ""
if args.predict_valid:
    args.predict = True
    args.valid_purchase = ""
extra_files = []
if args.train_clickthrough:
    args.epochs = 4 if args.epochs > 4 else args.epochs
    args.sort_train_data = True
    args.enable_weighted_loss = False
    
if not args.test and not args.predict:
    args.train = True
else:
    args.train = False
print(args)

from pathlib import Path
import os
import numpy as np
import torch as th
from torch.utils.data import DataLoader, SequentialSampler
from src.utils.data.dataset import read_dataset, AugmentedDataset
from src.utils.data.collate import (seq_to_ccs_graph, collate_fn_factory_ccs, collate_fn_features)
from src.utils.train import TrainRunner
from src.models import SIHG4SR
from src.scripts.utils import *
from tqdm import tqdm

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
#device = 'cpu'
dataset_dir = Path(args.dataset_dir)

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

item_features = None
if args.enable_features_gnn:
    ### Train or load a seperate network for item with features ###
    item_features_path = os.path.join(args.dataset_dir, "item_features.csv")
    logging.info("Loading item features from {}".format(item_features_path))
    with Timer("loaded, took "):
        item_features_df = load_file(item_features_path)

    item_features_extra_path = os.path.join(current_path, "../item_features_extra.csv")
    if os.path.exists(item_features_extra_path):
        logging.info("Loading item features extra from {}".format(item_features_extra_path))
        with Timer("loaded, took "):
            item_features_extra_df = load_file(item_features_extra_path)
            if 'itemcat_fp' in args.extra_feat_key:
                print(f"Bin itemcat to {args.itemcat_bin_size} categories")
                item_features_extra_df = bin_item_category(item_features_extra_df, num_cate = args.itemcat_bin_size)
    else:
        item_features_extra_df = None
        exclude_feat_ids = []

    #load feature category csv
    item_features_cate_path = os.path.join(current_path, "../categorical_item_features.csv")
    if os.path.exists(item_features_cate_path):
        logging.info("Loading item features from {}".format(item_features_cate_path))
        with Timer("loaded, took "):
            categorical_item_features_df = load_file(item_features_cate_path)
            exclude_feat_ids = get_exclude_feat_list(categorical_item_features_df, args.exclude_feat_id)
            print(f"exclude_feat_ids is {exclude_feat_ids}")
    else:
        categorical_item_features_df = None

else:
    item_features_df = None

    ################################################################

print('reading dataset')

candidate_path = os.path.join(args.dataset_dir, args.candidate_list)
logging.info("Loading candidate list from {}".format(candidate_path))
with Timer("loaded, took "):
    candidate_list = load_file(candidate_path)['item_id'].tolist()

try:
    if args.predict or args.test:
        raise FileNotFoundError("run predict, skip preload data")
    logging.info("Loading processed train and test data")
    with Timer("loaded, took "):
        train_set, num_unique_features = load_preprocessed(os.path.join(args.dataset_dir, "train_processed.parquet"), item_features_extra_df, recent_n_month = args.use_recent_n_month, candidate_list = None, add_features=args.enable_features_gnn, sort = args.sort_train_data, train_click = args.train_clickthrough, get_session_under_5 = args.session_under_5, get_session_above_5 = args.session_above_5, extra_feat_key = args.extra_feat_key, exclude_feat_ids = exclude_feat_ids)
        test_set, num_unique_features = load_preprocessed(os.path.join(args.dataset_dir, "test_processed.parquet"), item_features_extra_df, add_features=args.enable_features_gnn, get_session_under_5 = args.session_under_5, get_session_above_5 = args.session_above_5, extra_feat_key = args.extra_feat_key, predict=args.predict, exclude_feat_ids = exclude_feat_ids)
        num_items = 28144
except FileNotFoundError as e:
    # read from csv
    logging.info(f"{e}, Loading originl train data")
    if args.train:
        with Timer("loaded, took "):
            train_session = os.path.join(args.dataset_dir, args.train_session)
            train_target = os.path.join(args.dataset_dir, args.train_purchase)
            train_set, num_unique_features = preprocess(train_session, item_features_df, item_features_extra_df, train_target, recent_n_month = args.use_recent_n_month, add_features=args.enable_features_gnn, save_path = f"{args.dataset_dir}/train_processed.parquet", enable_weighted_loss = True, sort = args.sort_train_data, train_click = args.train_clickthrough, get_session_under_5 = args.session_under_5, get_session_above_5 = args.session_above_5, extra_feat_key = args.extra_feat_key, exclude_feat_ids = exclude_feat_ids)
    logging.info("Loading test data")
    with Timer("loaded, took "):
        valid_session = os.path.join(args.dataset_dir, args.valid_session)
        valid_target = os.path.join(args.dataset_dir, args.valid_purchase) if args.valid_purchase != "" else None
        test_set, num_unique_features = preprocess(valid_session, item_features_df, item_features_extra_df, valid_target, add_features=args.enable_features_gnn, save_path = f"{args.dataset_dir}/test_processed.parquet", enable_weighted_loss = True, get_session_under_5 = args.session_under_5, get_session_above_5 = args.session_above_5, extra_feat_key = args.extra_feat_key, exclude_feat_ids = exclude_feat_ids)
    num_items = 28144
except Exception as e:
    raise e

if args.prepare_data:
    exit()

logit_difference = [100 if i in candidate_list else 0 for i in range(num_items)]
logit_difference_batch = np.array([logit_difference for i in range(args.batch_size)])

### Do data augment ###
if args.data_augment > 1 and args.train:
    train_set = AugmentedDataset(train_set, args.data_augment)
if args.test_with_random_sample:
    test_set  = AugmentedDataset(test_set, 1)
### Do data augment ###

collate_fn = collate_fn_factory_ccs((seq_to_ccs_graph, ), order=args.order, attent_longest_view = args.attent_longest_view)

if args.train:
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        # shuffle=True,
        # drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=SequentialSampler(train_set))
else:
    train_loader = None

test_loader = DataLoader(test_set,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,
                         collate_fn=collate_fn,
                         pin_memory=True)

model = SIHG4SR(num_items,
                args.dataset_dir,
                args.embedding_dim,
                args.num_layers,
                dropout=args.feat_drop,
                reducer=args.reducer,
                order=args.order,
                norm=args.norm,
                extra=args.extra,
                fusion=args.fusion,
                device=device,
                enable_features_gnn=args.enable_features_gnn,
                num_unique_features = num_unique_features,
                enable_transformer=args.enable_transformer,
                transformer_head=args.transformer_head,
                transformer_layer=args.transformer_layer,
                srl_ratio=args.srl_ratio,
                srg_ratio=args.srg_ratio)

model = model.to(device)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

print(model)

if not args.resume_train and not args.predict and not args.test:
    args.model = ""
    
runner = TrainRunner(args.dataset_dir,
                     model,
                     train_loader,
                     test_loader,
                     device=device,
                     resumed_model = args.model,
                     save_path=args.save_path,
                     lr=args.lr,
                     weight_decay=args.weight_decay,
                     patience=args.patience,
                     topk=args.topk,
                     filter = logit_difference_batch,
                     weighted_loss = args.enable_weighted_loss,
                     enable_save_logits = args.enable_save_logits,
                     session_under_5 = args.session_under_5,
                     session_above_5 = args.session_above_5,
                     finetune = args.finetune
                     )
if args.test:
    mrr, hit = runner.evaluate(model)
    print('Test:')
    print(f'\tRecall@{args.topk}:\t{hit:.6f}\tMMR@{args.topk}:\t{mrr:.6f}\t')
    exit()

if args.predict:
    runner.predict(model)
    print(f'Predict completed, file is saved to {args.save_path}')
    exit()

print('start training', flush=True)
mrr, hit = runner.train(args.epochs, args.log_interval)
print(f'MRR@{args.topk}\tHR@{args.topk}', flush=True)
print(f'{mrr * 100:.3f}%\t{hit * 100:.3f}%', flush=True)
# add for hpo extractor
print(f'Final training result: MRR@{args.topk}:{mrr:.6f}, HR@{args.topk}:{hit:.6f}.', flush=True)
with open(os.path.join(args.save_path, "result_mrr"), "w") as f:
    f.write(f"{mrr:.6f}")
