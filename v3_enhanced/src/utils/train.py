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
    inputs, labels, sesses, wf, seqs = batch
    # inputs, labels = batch
    inputs_gpu = [x.to(device) for x in inputs]
    labels_gpu = labels.to(device) if labels is not None else None
    wf_gpu = wf.to(device) if wf is not None else 0

    return inputs_gpu, labels_gpu, sesses, wf_gpu, seqs
    # return inputs_gpu, 0, labels_gpu, 0

def save_predictions(preds, save_path, file_tail=""):
    import csv
    with open(f'{save_path}/prediction{file_tail}.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(["session_id", "item_id", "rank"])
        for sid, item_list in preds:
            for idx, iid in enumerate(item_list):
                spamwriter.writerow([sid, iid, idx + 1])
    print(f"{save_path}/prediction{file_tail}.csv is saved")

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
    

def predict(model, data_loader, device, save_path = "model_save", cutoff=20, filter = None, enable_save_logits=True, file_tail=""):
    model.eval()
    mrr = 0
    hit = 0
    num_samples = 0
    preds = []
    logits_to_save = []

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
            preds += current_preds
    save_predictions(preds, save_path, file_tail)
    if enable_save_logits:
        save_logits(logits_to_save, save_path)

def evaluate(model, data_loader, device, save_path = "model_save", cutoff=20, filter = None, enable_save_logits=False, file_tail=""):
    model.eval()
    mrr = 0
    hit = 0
    num_samples = 0
    preds = []
    logits_to_save = []

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

            preds += current_preds
            labels = labels.unsqueeze(-1)
            hit_ranks = th.where(topk == labels)[1] + 1
            hit += hit_ranks.numel()
            mrr += hit_ranks.float().reciprocal().sum().item()
            pbar.set_description(f"mrr is {(mrr * 1.0 /num_samples):.3f}")
    save_predictions(preds, save_path, file_tail)
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
                 resumed_model = "",
                 save_path="model_save",
                 lr=1e-3,
                 weight_decay=0,
                 patience=3,
                 topk=100,
                 filter=None,
                 feature_table=None,
                 weighted_loss=True,
                 enable_save_logits=False,
                 session_under_5 = False,
                 session_above_5 = False,
                 finetune = False):
        self.dataset = dataset

        if weight_decay > 0:
            params = fix_weight_decay(model)
        else:
            params = model.parameters()

        if finetune:
            self.optimizer = optim.SGD(params,
                                       lr=0.00910914690922397,
                                       momentum=0.9435342448267561,
                                       weight_decay = 0.0003233597640684583)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=3,
                                                   gamma=0.5)
        else:
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience
        self.topk = topk
        self.filter = th.Tensor(filter).to(device)
        self.save_path = save_path
        self.weighted_loss = weighted_loss
        self.enable_save_logits = enable_save_logits
        if session_above_5:
            self.tail = "_sessgr5"
        elif session_under_5:
            self.tail = "_sessls5"
        else:
            self.tail = ""

        if resumed_model != "":
            checkpoint = th.load(resumed_model)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                if not finetune:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.epoch = checkpoint['epoch']
            else:
                self.model.load_state_dict(checkpoint)
        self.finetune = finetune

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
                            filter = self.filter,
                            file_tail = self.tail)
        '''
        for epoch in tqdm(range(epochs)):
            self.model.train()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for idx, batch in pbar:
                inputs, labels, _, wf, _ = prepare_batch(batch, self.device)
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
                pbar.set_description(f"loss is {loss.item():.4f}")

                if self.batch > 0 and self.batch % log_interval == 0:
                    #print(
                    #    f'Batch {self.batch}: Loss = {mean_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s'
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
                                filter = self.filter,
                                file_tail = self.tail)

            print(
                f'Epoch {self.epoch}: MRR = {mrr * 100:.3f}%, Hit = {hit * 100:.3f}%'
            )

            if epoch != None:
                if not os.path.exists(self.save_path):
                    os.mkdir(self.save_path)
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}
                th.save(state, f"{self.save_path}/model_{int(mrr * 10000)}_{'finetune' if self.finetune else 'epoch'}_{epoch}{self.tail}.pth")

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

    def predict(self, model):
        return predict(model, self.test_loader, self.device, save_path=self.save_path, cutoff=self.topk, filter = self.filter, enable_save_logits=self.enable_save_logits, file_tail = self.tail)

    def evaluate(self, model):
        return evaluate(model, self.test_loader, self.device, save_path=self.save_path, cutoff=self.topk, filter = self.filter, enable_save_logits=self.enable_save_logits, file_tail = self.tail)