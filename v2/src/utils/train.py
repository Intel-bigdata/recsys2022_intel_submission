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
import time

import numpy as np
import torch as th
from sklearn.metrics import accuracy_score
from torch import nn, optim
from tqdm import tqdm
import itertools
import pandas as pd


# ignore weight decay for parameters in bias, batch norm and activation
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def prepare_batch(batch, device):
    inputs, labels = batch
    inputs_gpu  = [x.to(device) for x in inputs]
    labels_gpu  = labels.to(device)
   
    return inputs_gpu, labels_gpu


def load_model_from_dir(model_dir, model, optimizer, is_finetune):
    epoch = -1
    exist_model_path = sorted(
        model_dir.glob('*.pkl'), 
        key=lambda x: x.stat().st_mtime, 
        reverse=True
    )

    if exist_model_path:
        model_path = exist_model_path[0]
        print(f'load exist model from {model_path}...')
        checkpoint = th.load(model_path)

        model.load_state_dict(checkpoint['state_dict'])

        if not is_finetune:
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
        
    return model, optimizer, epoch


def save_model(model_path, model, epoch, optimizer):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    th.save(state, model_path)


class TrainRunner:
    def __init__(
        self,
        model_dir,
        model,
        train_loader=None,
        valid_loader=None,
        is_finetune=True,
        device=th.device('cpu'),
        lr=1e-3,
        weight_decay=0,
        patience=3,
    ):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if weight_decay > 0:
            params = fix_weight_decay(model)
        else:
            params = model.parameters()
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

        self.model, self.optimizer, self.epoch = load_model_from_dir(
            model_dir, model, self.optimizer, is_finetune
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=6, gamma=0.1, last_epoch=self.epoch
        )

        self.epoch += 1
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device       = device
        self.batch        = 0
        self.patience     = patience


    def train(self, epochs, candidate_nid, log_interval=100):
        max_mrr = 0
        max_hit = 0
        bad_counter = 0
        t = time.time()
        mean_loss = 0

        mrr, hit = 0, 0
        for epoch in tqdm(range(epochs)):
            self.model.train()
            for batch in self.train_loader:
                inputs, labels = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()
                scores = self.model(*inputs)
                assert not th.isnan(scores).any()
                loss   = nn.functional.nll_loss(scores, labels)
                loss.backward()
                self.optimizer.step()
                
                mean_loss += loss.item() / log_interval
                
                if self.batch > 0 and self.batch % log_interval == 0:
                    print(f'Batch {self.batch}: Loss = {mean_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s')
                    t = time.time()
                    mean_loss = 0
                    
                self.batch += 1
            self.scheduler.step()
            mrr, hit, valid_loss = self.evaluate(self.valid_loader, candidate_nid)
            
            print(f'Epoch {epoch+1}: MRR = {mrr * 100:.3f}%, Hit = {hit * 100:.3f}%, Loss = {valid_loss:.4f}')

            if mrr < max_mrr and hit < max_hit:
                bad_counter += 1
                if bad_counter == self.patience:
                    break
            else:
                bad_counter = 0
            max_mrr = max(max_mrr, mrr)
            max_hit = max(max_hit, hit)

            model_save_path = f'{self.model_dir}/net_parameter-{self.epoch+1}-{mrr * 100:.3f}.pkl'
            save_model(model_save_path, self.model, self.epoch, self.optimizer)
            self.epoch += 1

        return max_mrr, max_hit


    def predict(self, data_loader, candidate_nid, candidate_index2oid, cutoff=100):
        self.model.eval()
        session_id = []
        rank = []
        item_id = []

        with th.no_grad():
            for batch in data_loader:
                inputs, labels = prepare_batch(batch, self.device)
                logits = self.model(*inputs)
            
                session_id.append(np.repeat(labels.cpu().numpy(), cutoff))

                batch_size = logits.size(0)
                rank.append(np.tile(np.arange(1,cutoff+1), batch_size))

                topk = logits[:, candidate_nid].topk(k=cutoff)[1]
                item_id += list(
                    map(lambda i: pd.Series(i).map(candidate_index2oid).to_numpy(), topk.cpu().numpy())
                )

        session_id = itertools.chain.from_iterable(session_id)
        session_id = np.fromiter(session_id, dtype=np.long)

        rank = itertools.chain.from_iterable(rank)
        rank = np.fromiter(rank, dtype=np.long)

        item_id = itertools.chain.from_iterable(item_id)
        item_id = np.fromiter(item_id, dtype=np.long)

        return pd.DataFrame({
            'session_id': session_id,
            'item_id': item_id,
            'rank': rank,
        })
                
                
    def evaluate(self, data_loader, candidate_nid, cutoff=100):
        self.model.eval()
        mrr = 0
        hit = 0
        mean_loss = 0
        num_batchs = 0
        num_samples = 0

        with th.no_grad():
            for batch in data_loader:
                inputs, labels = prepare_batch(batch, self.device)
                logits = self.model(*inputs)
                assert not th.isnan(logits).any()
                loss   = nn.functional.nll_loss(logits, labels)
                mean_loss += loss.item()

                batch_size   = logits.size(0)
                num_samples += batch_size
                mask = th.zeros(logits.shape).to(self.device)
                mask[:, candidate_nid] = 1
                logits = logits.masked_fill(~mask.bool(), float('-inf'))
                topk         = logits.topk(k=cutoff)[1]
                labels       = labels.unsqueeze(-1)
                hit_ranks    = th.where(topk == labels)[1] + 1
                hit         += hit_ranks.numel()
                mrr         += hit_ranks.float().reciprocal().sum().item()
                num_batchs  += 1
                
        return mrr / num_samples, hit / num_samples, mean_loss / num_batchs
