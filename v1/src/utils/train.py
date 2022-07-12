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
from pandas import cut
import torch as th
from sklearn.metrics import accuracy_score
from torch import nn, optim
from tqdm import tqdm
import os
from sklearn.neighbors import KNeighborsClassifier

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
    # if len(batch) == 4:
    #     inputs, labels, sesses, wf = batch
    # else:
    #     inputs, labels = batch
    #     sesses = None
    inputs, labels, sesses, wf, seqs = batch
    # inputs, labels = batch
    inputs_gpu = [x.to(device) for x in inputs]
    labels_gpu = labels.to(device) if labels is not None else None
    wf_gpu = wf.to(device) if wf is not None else 0

    return inputs_gpu, labels_gpu, sesses, wf_gpu, seqs
    # return inputs_gpu, 0, labels_gpu, 0

def save_predictions(preds, save_path):
    import csv
    with open(f'{save_path}/prediction.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(["session_id", "item_id", "rank"])
        for sid, item_list in preds:
            for idx, iid in enumerate(item_list):
                spamwriter.writerow([sid, iid, idx + 1])
    print(f"{save_path}/prediction.csv is saved")

def save_logits(logits_to_save, save_path):
    import pandas as pd
    if len(logits_to_save[0]) == 2:
        pdf = pd.DataFrame(logits_to_save, columns =['session_id', 'logits'])
        pdf.to_parquet(f"{save_path}/logits.parquet")
    elif len(logits_to_save[0]) == 3:
        pdf = pd.DataFrame(logits_to_save, columns =['session_id', 'logits', 'labels'])
        if pdf.shape[0] > 40000:
            pdf[:40000].to_parquet(f"{save_path}/logits1.parquet")
            pdf[40000:].to_parquet(f"{save_path}/logits2.parquet")
    print(f"{save_path}/logits.parquet is saved")

def do_append_similiar_item(preds, feature_model, feature_table, neigh):
    append_pres = [[], [], [], []]
    input = feature_table[preds]
    if feature_model:
        input_tensor = th.LongTensor(input)
        logits = feature_model(input_tensor)
        _, topk = logits.topk(4)
        topk = topk.detach().tolist()
    elif neigh:
        _, topk = neigh.kneighbors(input, n_neighbors=5)
    for pred, iids in zip(preds, topk):
        for i in range(4):
            append_pres[i].append(iids[i+1])
    return preds + append_pres[0] + append_pres[1] + append_pres[2] + append_pres[3]
    

def predict(model, data_loader, device, save_path = "model_save", cutoff=20, filter = None, enable_features = False, feature_table = None, feature_model = None,enable_save_logits=True):
    model.eval()
    mrr = 0
    hit = 0
    num_samples = 0
    preds = []
    logits_to_save = []
    append_similiar_item = False
    neigh = None 
    if enable_features:
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(feature_table, range(1, len(feature_table) + 1))
        cutoff = int(cutoff / 5)
        append_similiar_item = True

    with th.no_grad():
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for idx, batch in pbar:
            inputs, _, sesses, _, seqs = prepare_batch(batch, device)
            logits = model(*inputs)
            if enable_save_logits:
                logits_to_save += [[sid, logit] for sid, logit in zip(sesses, logits.detach().tolist())]
            batch_size = logits.size(0)
            num_items = logits.size(1)
            # apply filter #
            filter_exclude = th.zeros([batch_size, num_items]).to(device)
            for i, seq in enumerate(seqs):
                filter_exclude[i][seq] += -100
            logits += filter[:batch_size]
            logits += filter_exclude[:batch_size]
            ################
            num_samples += batch_size
            _, topk = logits.topk(cutoff)
            current_preds = [[sid, iid] for sid, iid in zip(sesses, topk.detach().tolist())]
            if append_similiar_item:
                new_preds = []
                for sid, item_list in current_preds:                    
                    new_preds.append([sid, do_append_similiar_item(item_list, feature_model, feature_table, neigh)])
                current_preds = new_preds
            preds += current_preds
    save_predictions(preds, save_path)
    if enable_save_logits:
        save_logits(logits_to_save, save_path)

def evaluate(model, data_loader, device, save_path = "model_save", cutoff=20, filter = None, enable_features = False, feature_table = None, feature_model = None,enable_save_logits=False):
    model.eval()
    mrr = 0
    hit = 0
    num_samples = 0
    preds = []
    logits_to_save = []
    append_similiar_item = False
    neigh = None 
    if enable_features:
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(feature_table, range(1, len(feature_table) + 1))
        cutoff = int(cutoff / 5)
        append_similiar_item = True

    with th.no_grad():
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for idx, batch in pbar:
            inputs, labels, sesses, _, seqs = prepare_batch(batch, device)
            logits = model(*inputs)
            if enable_save_logits:
                logits_tmp = logits.detach().tolist()
                labels_tmp = labels.detach().tolist()
                for sid in range(len(sesses)):
                    logits_to_save.append([sesses[sid], logits_tmp[sid], labels_tmp[sid]])
            batch_size = logits.size(0)
            num_items = logits.size(1)
            # apply filter #
            filter_exclude = th.zeros([batch_size, num_items]).to(device)
            for i, seq in enumerate(seqs):
                filter_exclude[i][seq] += -100
            logits += filter[:batch_size]
            logits += filter_exclude[:batch_size]
            ################
            num_samples += batch_size
            _, topk = logits.topk(cutoff)
            current_preds = [[sid, iid] for sid, iid in zip(sesses, topk.detach().tolist())]
            if append_similiar_item:
                new_preds = []
                for sid, item_list in current_preds:                    
                    new_preds.append([sid, do_append_similiar_item(item_list, feature_model, feature_table, neigh)])
                current_preds = new_preds
            preds += current_preds
            labels = labels.unsqueeze(-1)
            hit_ranks = th.where(topk == labels)[1] + 1
            hit += hit_ranks.numel()
            mrr += hit_ranks.float().reciprocal().sum().item()
            pbar.set_description(f"mrr is {(mrr * 1.0 /num_samples):.5f}")
    save_predictions(preds, save_path)
    if enable_save_logits:
        save_logits(logits_to_save, save_path)

    return mrr / num_samples, hit / num_samples



class TrainRunner:

    def __init__(self,
                 dataset,
                 model,
                 train_loader,
                 test_loader,
                 device,
                 save_path="model_save",
                 lr=1e-3,
                 weight_decay=0,
                 patience=3,
                 topk=100,
                 filter=None,
                 feature_table=None,
                 weighted_loss=True,
                 enable_save_logits=False,
                 step_size=3,
                 gamma=0.1,
                 optimizer_type="Adam",
                 momentum=0.9,
                 model_path=None):
        self.dataset = dataset
        self.model = model
        if weight_decay > 0:
            params = fix_weight_decay(model)
        else:
            params = model.parameters()
            
        if optimizer_type == "Adam":
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "SGD":
            self.optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == "Adadelta":
            self.optimizer = optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=step_size,
                                                   gamma=gamma)

        if model_path is not None:
            ckpt = th.load(model_path)
            self.model.load_state_dict(ckpt)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience
        self.topk = topk
        self.filter = th.Tensor(filter).to(device)
        self.feature_table = feature_table
        self.save_path = save_path
        self.weighted_loss = weighted_loss
        self.enable_save_logits = enable_save_logits

    def train(self, epochs, log_interval=100):
        max_mrr = 0
        max_hit = 0
        bad_counter = 0
        t = time.time()
        mean_loss = 0

        '''
        mrr, hit = evaluate(self.model,
                            self.test_loader,
                            self.device,
                            save_path=self.save_path,
                            cutoff=self.topk,
                            filter = self.filter)
        '''
        for epoch in tqdm(range(epochs)):
            self.model.train()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for idx, batch in pbar:
                inputs, labels, _, wf,_ = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()
                scores = self.model(*inputs)
                assert not th.isnan(scores).any()
                if self.weighted_loss:
                    loss = -th.gather(scores, dim=1, index=th.unsqueeze(labels, dim=1)).squeeze() * wf
                    loss = loss.sum() / wf.sum()
                else:
                    loss = nn.functional.nll_loss(scores, labels)
                loss.backward()
                self.optimizer.step()

                mean_loss += loss.item() / log_interval
                pbar.set_description(f"loss is {loss.item():.5f}")

                if self.batch > 0 and self.batch % log_interval == 0:
                    #print(
                    #    f'Batch {self.batch}: Loss = {mean_loss:.5f}, Time Elapsed = {time.time() - t:.2f}s'
                    #)
                    t = time.time()
                    mean_loss = 0

                self.batch += 1
            self.scheduler.step()
            mrr, hit = evaluate(self.model,
                                self.test_loader,
                                self.device,
                                save_path=self.save_path,
                                cutoff=self.topk,
                                filter = self.filter)

            print(
                f'Epoch {self.epoch}: MRR = {mrr * 100:.5f}%, Hit = {hit * 100:.5f}%'
            )
            if epoch != None:
                if not os.path.exists(self.save_path):
                    os.mkdir(self.save_path)
                th.save(self.model.state_dict(), f"{self.save_path}/model_{epoch}_{mrr:.5f}.pth")

            if mrr < 0.1 or (mrr <= max_mrr and hit <= max_hit) :
                bad_counter += 1
                if bad_counter == self.patience:
                    break
            else:
                bad_counter = 0
            max_mrr = max(max_mrr, mrr)
            max_hit = max(max_hit, hit)
            self.epoch += 1
        return max_mrr, max_hit

    def predict(self, model, enable_features = False, feature_model = None):
        return predict(model, self.test_loader, self.device, save_path=self.save_path, cutoff=self.topk, filter = self.filter, enable_features = enable_features, feature_table = self.feature_table, feature_model = feature_model,enable_save_logits=self.enable_save_logits)

    def evaluate(self, model, enable_features = False, feature_model = None):
        return evaluate(model, self.test_loader, self.device, save_path=self.save_path, cutoff=self.topk, filter = self.filter, enable_features = enable_features, feature_table = self.feature_table, feature_model = feature_model,enable_save_logits=self.enable_save_logits)