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


import math

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import dgl.ops as F
import numpy as np
import torch as th
import torch.nn as nn
from scipy import sparse

from .gnn_models import GATConv
#from dgl.nn.pytorch.conv import GATConv

class SemanticExpander(nn.Module):
    
    def __init__(self, input_dim, reducer, order):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.order = order
        self.reducer = reducer
        self.GRUs = nn.ModuleList()
        for i in range(self.order):
            self.GRUs.append(nn.GRU(self.input_dim, self.input_dim, 1, True, True))
    
        if self.reducer == 'concat':
            self.Ws = nn.ModuleList()
            for i in range(1, self.order):
                self.Ws.append(nn.Linear(self.input_dim * (i+1), self.input_dim))
        
    def forward(self, feat):
        
        if len(feat.shape) < 3:
            return feat
        if self.reducer == 'mean':
            invar = th.mean(feat, dim=1)
        elif self.reducer == 'max':
            invar =  th.max(feat, dim=1)[0]
        elif self.reducer == 'concat':
            invar =  self.Ws[feat.size(1)-2](feat.view(feat.size(0), -1))
        var = self.GRUs[feat.size(1)-2](feat)[1].permute(1, 0, 2).squeeze()

        # return invar + var
        return 0.5 * invar + 0.5 * var
      
class MSHGNN(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout=0.0, activation=None, order=1, reducer='mean', enable_features_gnn = False):
        super().__init__()
     
        self.dropout = nn.Dropout(dropout)
        # self.gru = nn.GRUCell(2 * input_dim, output_dim)
        self.output_dim = output_dim
        self.activation = activation
        self.order = order

        '''
        conv1_modules = {}
        ''' 
        conv1_modules = {'intra'+str(i+1) : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True) for i in range(self.order)}
        conv1_modules.update({'inter'     : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True)})
        if enable_features_gnn:
            conv1_modules.update({'attr'     : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True)})
            conv1_modules.update({'cat'     : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True)})
        self.conv1 = dglnn.HeteroGraphConv(conv1_modules, aggregate='sum')
        
        '''
        conv2_modules = {}
        '''
        conv2_modules = {'intra'+str(i+1) : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True) for i in range(self.order)}
        conv2_modules.update({'inter'     : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True)})
        if enable_features_gnn:
            conv2_modules.update({'attr'     : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True)})
            conv2_modules.update({'cat'     : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True)})
        self.conv2 = dglnn.HeteroGraphConv(conv2_modules, aggregate='sum')
        
        self.lint = nn.Linear(output_dim, 1, bias=False)
        self.linq = nn.Linear(output_dim, output_dim)
        self.link = nn.Linear(output_dim, output_dim, bias=False)
        
    def forward(self, g, feat):
        
        with g.local_scope():
                
            h1 = self.conv1(g, (feat, feat))
            #h2 = self.conv2(g, (feat, feat))
            h2 = self.conv2(g.reverse(copy_edata=True), (feat, feat))
            h = {}
            h['f1'] = self.post_process(g, feat, h1, h2, 'f1')
            h['c1'] = self.post_process(g, feat, h1, h2, 'c1')
            for i in range(self.order):
                h['s'+str(i+1)] = self.post_process(g, feat, h1, h2, 's'+str(i+1))
        return h

    def post_process(self, g, feat, h1, h2, key):
        hfl, hfr = th.zeros(1, self.output_dim).to(self.lint.weight.device), th.zeros(1, self.output_dim).to(self.lint.weight.device)
        if key in h1:
            hfl = h1[key]
        if key in h2:
            hfr = h2[key]
        h = hfl + hfr
        if len(h.shape) > 2:
            h = h.max(1)[0]
        h_mean = F.segment.segment_reduce(g.batch_num_nodes(key), feat[key], 'mean')
        h_mean = dgl.broadcast_nodes(g, h_mean, ntype=key) # map per batch value to all nodes
        h = h + h_mean
        return h
                   
class AttnReadout(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        feat_drop=0.0,
        activation=None,
        order=1,
        device=th.device('cpu'),
        features_as_s2 = False
    ):
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop)
        self.order = order
        if features_as_s2:
            self.order = 2
        self.order_range = range(order) if not features_as_s2 else [0, 1]
        self.order_range_g = range(order) if not features_as_s2 else [0, 0]
        self.device = device
        self.fc_u = nn.ModuleList()
        self.fc_v = nn.ModuleList()
        self.fc_e = nn.ModuleList()
        self.fc_p = nn.ModuleList()
        for i in range(self.order):
            self.fc_u.append(nn.Linear(input_dim, hidden_dim, bias=True))
            self.fc_v.append(nn.Linear(input_dim, hidden_dim, bias=False))
            self.fc_e.append(nn.Linear(hidden_dim, 1, bias=False))
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim
            else None
        )
        self.activation = activation
        
    def forward(self, g, feats, last_nodess):
        
        rsts = []
      
        nfeats = []
        for i, j in zip(self.order_range, self.order_range_g): 
            feat = feats['s'+str(i+1)]
            feat = th.split(feat, g.batch_num_nodes('s'+str(j+1)).tolist())
            feats['s'+str(i+1)] = th.cat(feat, dim=0)
            nfeats.append(feat)
        feat_vs= th.cat(tuple(feats['s'+str(i+1)][last_nodess[i]].unsqueeze(1) for i in self.order_range), dim=1)
        feats = th.cat([th.cat(tuple(nfeats[j][i] for j in self.order_range), dim=0) for i in range(len(g.batch_num_nodes('s1')))], dim=0)
        batch_num_nodes = th.cat(tuple(g.batch_num_nodes('s'+str(i+1)).unsqueeze(1) for i in self.order_range_g), dim=1).sum(1)
       
        idx = th.cat(tuple(th.ones(batch_num_nodes[j])*j for j in range(len(batch_num_nodes)))).long()
        for i in self.order_range:
            feat_u = self.fc_u[i](feats) 
            feat_v = self.fc_v[i](feat_vs[:, i])[idx]
            e = self.fc_e[i](th.sigmoid(feat_u + feat_v))
            alpha = F.segment.segment_softmax(batch_num_nodes, e)
            
            feat_norm = feats * alpha
            rst = F.segment.segment_reduce(batch_num_nodes, feat_norm, 'sum')
            rsts.append(rst.unsqueeze(1))
        
            if self.fc_out is not None:
                rst = self.fc_out(rst)
            if self.activation is not None:
                rst = self.activation(rst)
        rst = th.cat(rsts, dim=1)
        
        return rst

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, d_ff=512, dropout = 0.1):
        super().__init__() 
        self.linear_1 = nn.Linear(embedding_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, embedding_dim)
    def forward(self, x):
        x = self.dropout(nn.functional.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, transformer_head, device, dropout=0.1):
        super().__init__()
        self.device = device
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=transformer_head, dropout=0.1)
        self.ff = FeedForward(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.layer_norm1(x)
        x2, _ = self.attn(x2, x2, x2, attn_mask=get_mask(x2.shape[0], self.device))
        x = x + x2
        x2 = self.layer_norm2(x)
        x = x + self.dropout(self.ff(x2))
        return x

class SIHG4SR(nn.Module):
    
    def __init__(self, num_items, datasets, embedding_dim, num_layers, dropout=0.0, reducer='mean', order=3, norm=True, extra=True, fusion=True, device=th.device('cpu'), enable_features_gnn = False, num_unique_features = -1, enable_transformer=False, transformer_head=2, transformer_layer=1,srl_ratio=0.7,srg_ratio=0.3):
        super().__init__()
        
        self.embeddings = nn.Embedding(num_items, embedding_dim, max_norm=1)
        self.features_embeddings = nn.Embedding(num_unique_features, embedding_dim, max_norm=1) if num_unique_features != -1 else None
        self.category_embeddings = nn.Embedding(74, embedding_dim, max_norm=1) if num_unique_features != -1 else None
        self.num_unique_features = num_unique_features
        
 
        self.num_items = num_items
        self.register_buffer('indices', th.arange(num_items, dtype=th.long))
        self.register_buffer('feature_indices', th.arange(num_unique_features, dtype=th.long))
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.layers   = nn.ModuleList()
        input_dim     = embedding_dim
        self.reducer  = reducer
        self.order    = order
        self.alpha    = nn.Parameter(th.Tensor(self.order))
        self.beta     = nn.Parameter(th.Tensor(1))
        self.norm     = norm
        self.expander = SemanticExpander(input_dim, reducer, order)
        self.enable_transformer = enable_transformer
        self.transformer_head = transformer_head
        self.transformer_layer = transformer_layer
        self.device = device
        self.srl_ratio = srl_ratio
        self.srg_ratio = srg_ratio
        for i in range(num_layers):
            layer = MSHGNN(
                input_dim,
                embedding_dim,
                dropout=dropout,
                order=self.order,
                activation=nn.PReLU(embedding_dim),
                enable_features_gnn = enable_features_gnn
            )
            self.layers.append(layer)
            
        self.readout = AttnReadout(
            input_dim,
            embedding_dim,
            embedding_dim,
            feat_drop=dropout,
            activation=None,
            order=self.order,
            device=self.device,
            features_as_s2 = self.features_embeddings != None,
        )
        input_dim += embedding_dim
        self.feat_drop = nn.Dropout(dropout)

        if self.enable_transformer:
            print(f"add Transformer with layer {self.transformer_layer} and head {self.transformer_head}")
            self.attn_layers = nn.ModuleList()
            for i in range(self.transformer_layer):
                self.attn_layers.append(EncoderLayer(embedding_dim, self.transformer_head, self.device))

        self.sc_sr = nn.ModuleList()
        for i in range(self.order):
            self.sc_sr.append(nn.Sequential(nn.Linear(embedding_dim, embedding_dim, bias=True),  nn.ReLU(), nn.Linear(embedding_dim, 2, bias=False), nn.Softmax(dim=-1)))
        self.input_dim = input_dim
        self.embedding_dim =embedding_dim
 
        # self.sr_trans1 = nn.Linear(embedding_dim, embedding_dim)
        # self.sr_trans2 = nn.Linear(embedding_dim, embedding_dim)
        self.reset_parameters()
        self.alpha.data = th.zeros(self.order)
        self.alpha.data[0] = th.tensor(1.0)
        # self.beta.data = th.zeros(1)
        self.beta.data = th.tensor(1.0)
        self.fusion = fusion
        self.extra = extra
        self.epoch = 0
        
    def inc_epoch(self):
        self.epoch += 1
          
    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def iid2rp(self, iid):
        tmp = th.sum(th.cat(tuple(th.unique(tmp, return_inverse=True)[1].unsqueeze(0) for tmp in th.unbind(iid, dim=0)), dim=0), dim=1)
        
        return tmp
        
    def residual(self, h1, res):
        
        for key in h1.keys():
            h1[key] += res[key]
        
        return h1

    def shuffle_to_item(self, mg, etype, h):
        hfl = []
        src_nodes, dst_nodes = mg.edges(etype = etype, order = 'srcdst')
        src_nodes = src_nodes.tolist()
        dst_nodes = dst_nodes.tolist()
        dst_agg_list = []
        cur_id = -1
        for src, dst in zip(src_nodes, dst_nodes):
            if src > cur_id:
                dst_agg_list.append([])
                cur_id = src
            dst_agg_list[-1].append(dst)
        
        for fl in dst_agg_list:
            h_f = th.mean(h[fl], 0, True)
            hfl.append(h_f)
        #print(hfl)
        return th.cat(hfl, dim = 0)

    def broadcast_to_item(self, mg, key, h):
        h_mean = F.segment.segment_reduce(mg.batch_num_nodes(key), h, 'mean')
        return dgl.broadcast_nodes(mg, h_mean, ntype='s1') # map per batch value to all nodes
        
    def forward(self, mg):
        
        ### GNN layer
        feats = {}
        for i in range(self.order):
            iid = mg.nodes['s' + str(i+1)].data['iid']
            feat = self.embeddings(iid) 
            feat = self.feat_drop(feat)
            feat = self.expander(feat)
            if th.isnan(feat).any():
                feat = feat.masked_fill(feat != feat, 0)
            if self.norm:
                feat = nn.functional.normalize(feat, dim=-1)
            feats['s' + str(i+1)] = feat

        if self.features_embeddings != None: 
            index = mg.nodes['f1'].data['fid']
            feats['f1'] = self.features_embeddings(index)
       
        if self.category_embeddings != None: 
            index = mg.nodes['c1'].data['cid']
            feats['c1'] = self.category_embeddings(index)
       
        h = feats
        for idx, layer in enumerate(self.layers):
            h = layer(mg, h)

        if self.enable_transformer:
            feat_tmp = h['f1']
            batch_num_nodes_lists = mg.batch_num_nodes('f1').tolist()
            max_n_node = np.max(batch_num_nodes_lists)
            start = 0
            feat_tmp_out = []
            for num_nodes in batch_num_nodes_lists:
                feat_tmp_per_batch = feat_tmp[start: (start + num_nodes)]
                end_zeros = th.from_numpy(np.zeros((max_n_node-num_nodes, feat_tmp_per_batch.shape[1]))).to(self.device)
                feat_tmp_out.append(th.cat([feat_tmp_per_batch, end_zeros],0).unsqueeze(0))
                start += num_nodes
            feat_tmp_out = th.cat(feat_tmp_out,0).float()

            feat_tmp_out = feat_tmp_out.permute(1, 0, 2)

            for idx, layer in enumerate(self.attn_layers):
                feat_tmp_out = layer(feat_tmp_out)

            feat_tmp_out = feat_tmp_out.permute(1, 0, 2)

            feat_tmp = []
            for i in range(feat_tmp_out.shape[0]):
                feat_tmp_sub = feat_tmp_out[i][:batch_num_nodes_lists[i]]
                if batch_num_nodes_lists[i] > 1:
                    feat_tmp_sub = feat_tmp_sub.squeeze(0)
                feat_tmp.append(feat_tmp_sub)
            h['f1'] = th.cat(feat_tmp, 0)

        # try this first, simply add features embedding back to items
        if self.features_embeddings != None:
            h['s2'] = self.shuffle_to_item(mg, 'attr', h['f1']) + self.broadcast_to_item(mg, 'c1', h['c1'])

        last_nodes = []
        for i in range(self.order):
            if self.norm:
                h['s'+str(i+1)] = nn.functional.normalize(h['s'+str(i+1)], dim=-1)
            last_nodes.append(mg.filter_nodes(lambda nodes: nodes.data['last'] == 1, ntype='s'+str(i+1)))
            
        if 's2' in h:
            # hack!!! use fusion to combine
            last_nodes.append(last_nodes[-1])
            order_range = [0, 1]
        else:
            order_range = range(self.order)

        feat = h
        ### attention, h is embedding table for items in one session
        sr_g = self.readout(mg, feat, last_nodes)                                                               

        sr_l = th.cat([feat['s'+str(i+1)][last_nodes[i]].unsqueeze(1) for i in order_range], dim=1)
        sr = self.srl_ratio * sr_l + self.srg_ratio * sr_g
        '''
        sr   = th.cat([sr_l, sr_g], dim=-1)# .view(sr_l.size(0), -1)
        sr   = th.cat([self.fc_sr[i](sr).unsqueeze(1) for i, sr in enumerate(th.unbind(sr, dim=1))], dim=1)
        '''
        # print(f"sr_l shape {sr_l.shape}, sr_g shape {sr_g.shape}, sr shape {sr.shape}")
        # exit()
        if self.norm:
            sr = nn.functional.normalize(sr, dim=-1)
        
        target = self.embeddings(self.indices)
        
        if self.norm:
            target = nn.functional.normalize(target, dim=-1)
               
        if self.extra:
            logits = sr @ target.t()
            phi = self.sc_sr[0](sr).unsqueeze(-1)
            mask = th.zeros(phi.size(0), self.num_items).to(self.device)
            iids = th.split(mg.nodes['s1'].data['iid'], mg.batch_num_nodes('s1').tolist())
            for i in range(len(mask)):
                mask[i, iids[i]] = 1

            logits_in = logits.masked_fill(~mask.bool().unsqueeze(1), float('-inf'))
            logits_ex = logits.masked_fill(mask.bool().unsqueeze(1), float('-inf'))
            score     = th.softmax(12 * logits_in.squeeze(), dim=-1)
            score_ex  = th.softmax(12 * logits_ex.squeeze(), dim=-1) 
          
            if th.isnan(score).any():
                score    = score.masked_fill(score != score, 0)
            if th.isnan(score_ex).any():
                score_ex = score_ex.masked_fill(score_ex != score_ex, 0)
            assert not th.isnan(score).any()
            assert not th.isnan(score_ex).any()
            # print(score.shape, score_ex.shape)
            if len(order_range) == 1:
                phi = phi.squeeze(1)
                score = (th.cat((score.unsqueeze(1), score_ex.unsqueeze(1)), dim=1) * phi).sum(1)
            else:
                score = (th.cat((score.unsqueeze(2), score_ex.unsqueeze(2)), dim=2) * phi).sum(2)
        else:
            # print("no extra ****************")
            logits = sr.squeeze() @ target.t()
            score  = th.softmax(12 * logits, dim=-1)
        
        if len(order_range) > 1:
            alpha = th.softmax(self.alpha.unsqueeze(0), dim=-1).view(1, self.alpha.size(0), 1)
            g = alpha.repeat(score.size(0), 1, 1)
            score = (score * g).sum(1)
            
        # print(score.shape)
            
        score = th.log(score)
        if th.isnan(score).any():
            score = score.masked_fill(score != score, 0)
        
        return score
        
        
def get_mask(seq_len,device):
    return th.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool')).to(device)
