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
import dgl.nn.pytorch as dglnn
import dgl.ops as F
import torch as th
import torch.nn as nn

from .gnn_models import GATConv
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence


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
        
    def forward(self, feat, graph_node=True, seq_len=None):
        
        if len(feat.shape) < 3:
            return feat
        if self.reducer == 'mean':
            invar = th.mean(feat, dim=1)
        elif self.reducer == 'max':
            invar =  th.max(feat, dim=1)[0]
        elif self.reducer == 'concat':
            invar =  self.Ws[feat.size(1)-2](feat.view(feat.size(0), -1))

        if graph_node:
            var = self.GRUs[feat.size(1)-2](feat)[1].permute(1, 0, 2).squeeze()
        else:
            feat = pack_padded_sequence(feat, seq_len, batch_first=True, enforce_sorted=False)
            var = self.GRUs[-1](feat)[1].permute(1, 0, 2).squeeze()
        return 0.5 * invar + 0.5 * var
        

class MSHGNN(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout=0.0, 
    device=th.device('cpu'),
    activation=None, order=1, message_func='Weight'):
        super().__init__()
     
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.output_dim = output_dim
        self.activation = activation
        self.order = order
        
        conv1_modules = {'intra'+str(i+1) : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True, activation=activation, message_func=message_func) for i in range(self.order)}
        conv1_modules.update({'inter'     : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True, activation=activation, message_func=message_func)})
        self.conv1 = dglnn.HeteroGraphConv(conv1_modules, aggregate='sum')
        
        conv2_modules = {'intra'+str(i+1) : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True, activation=activation, message_func=message_func) for i in range(self.order)}
        conv2_modules.update({'inter'     : GATConv(input_dim, output_dim, 8, dropout, dropout, residual=True, activation=activation, message_func=message_func)})
        self.conv2 = dglnn.HeteroGraphConv(conv2_modules, aggregate='sum')
                
    def forward(self, g, feat):
        
        with g.local_scope():
                
            h1 = self.conv1(g, (feat, feat))
            h2 = self.conv2(g.reverse(copy_edata=True), (feat, feat))
            h = {}
            for i in range(self.order):
                hl, hr = th.zeros(1, self.output_dim).to(self.device), th.zeros(1, self.output_dim).to(self.device)
                if 's'+str(i+1) in h1:
                    hl = h1['s'+str(i+1)]
                if 's'+str(i+1) in h2:
                    hr = h2['s'+str(i+1)]
                h['s'+str(i+1)] = hl + hr
                if len(h['s'+str(i+1)].shape) > 2:
                    h['s'+str(i+1)] = h['s'+str(i+1)].max(1)[0]
                h_mean = F.segment.segment_reduce(g.batch_num_nodes('s'+str(i+1)), feat['s'+str(i+1)], 'mean')
                h_mean = dgl.broadcast_nodes(g, h_mean, ntype='s'+str(i+1)) # adding mean maskes better
                h['s'+str(i+1)] =  h_mean + h['s'+str(i+1)]
                
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
        device=th.device('cpu')
    ):
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop)
        self.order = order
        self.device = device
        self.fc_u = nn.ModuleList()
        self.fc_v = nn.ModuleList()
        self.fc_e = nn.ModuleList()
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
        for i in range(self.order): 
            feat = feats['s'+str(i+1)]
            feat = th.split(feat, g.batch_num_nodes('s'+str(i+1)).tolist())
            feats['s'+str(i+1)] = th.cat(feat, dim=0)
            nfeats.append(feat)
        feat_vs= th.cat(tuple(feats['s'+str(i+1)][last_nodess[i]].unsqueeze(1) for i in range(self.order)), dim=1)
        feats = th.cat([th.cat(tuple(nfeats[j][i] for j in range(self.order)), dim=0) for i in range(len(g.batch_num_nodes('s1')))], dim=0)
        batch_num_nodes = th.cat(tuple(g.batch_num_nodes('s'+str(i+1)).unsqueeze(1) for i in range(self.order)), dim=1).sum(1)

        idx = th.cat(tuple(th.ones(batch_num_nodes[j])*j for j in range(len(batch_num_nodes)))).long()
        for i in range(self.order):
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


class TimeAttnReadout(nn.Module):
    def __init__(
        self,
        feat_input_dim,
        context_input_dim,
        hidden_dim,
        activation=None,
        device=th.device('cpu')
    ):
        super().__init__()
        self.device = device
        self.fc_u = nn.Linear(feat_input_dim, hidden_dim, bias=True)
        self.fc_v = nn.Linear(context_input_dim, hidden_dim, bias=False)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = activation
        
    def forward(self, feats, feat_context, batch_num_items):
        feat_u = self.fc_u(feats) 
        feat_v = self.fc_v(feat_context)
        e = self.fc_e(th.sigmoid(feat_u + feat_v))
        alpha = F.segment.segment_softmax(batch_num_items, e)
        
        feat_norm = feats * alpha
        rst = F.segment.segment_reduce(batch_num_items, feat_norm, 'sum')
        rst = self.fc_out(rst)
    
        if self.activation is not None:
            rst = self.activation(rst)
            
        return rst


def get_embedding_size_from_cardinality(cardinality, multiplier=5.0):
    # A rule-of-thumb from Google.
    embedding_size = int(math.floor(math.pow(cardinality, 0.25) * multiplier))
    return embedding_size


class SIHG4SR(nn.Module):
    
    def __init__(self, num_items, item_date_map, embedding_dim, num_layers, 
        dropout=0.0, reducer='mean', order=3, norm=True, extra=True, fusion=True, 
        device=th.device('cpu'), feat_length=905):
        super().__init__()
        
        self.embeddings = nn.Embedding(
            num_items+feat_length, embedding_dim, 
            max_norm=1, padding_idx=num_items
        )

        # item publish time embedding
        self.time_dim = 78
        self.time_embedding_dim = get_embedding_size_from_cardinality(self.time_dim)
        self.time_embeddings = nn.Embedding(self.time_dim, self.time_embedding_dim, max_norm=1)

        self.num_items = num_items
        self.register_buffer('indices', th.arange(num_items, dtype=th.long))
        self.register_buffer(
            'time_indices', 
            th.IntTensor([
                item_date_map.get(id, [0, 13, 45, 53])
                for id in range(self.num_items)
            ])
        )
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.layers   = nn.ModuleList()
        input_dim     = embedding_dim
        self.reducer  = reducer
        self.extra_order   = 1
        self.order    = order + self.extra_order
        self.alpha    = nn.Parameter(th.Tensor(self.order))
        self.norm     = norm
        self.expander = SemanticExpander(input_dim, reducer, self.order-self.extra_order)
        
        self.device = device
        self.feat_length = feat_length

        for i in range(num_layers):
            layer = MSHGNN(
                input_dim,
                embedding_dim,
                dropout=dropout,
                device=self.device,
                order=self.order,
                activation=nn.LeakyReLU(0.2)
            )
            self.layers.append(layer)
            
        self.readout = AttnReadout(
            input_dim,
            embedding_dim,
            embedding_dim,
            feat_drop=dropout,
            activation=None,
            order=self.order,
            device=self.device
        )

        self.time_readout = TimeAttnReadout(
            self.time_embedding_dim * 4,
            embedding_dim,
            self.time_embedding_dim * 4,
            activation=None,
            device=self.device
        )

        self.feat_drop = nn.Dropout(dropout)

        self.sc_sr = nn.ModuleList()
        for i in range(self.order):
            self.sc_sr.append(nn.Sequential(
                nn.Linear(embedding_dim + self.time_embedding_dim*4, embedding_dim, bias=True),  
                nn.ReLU(), 
                nn.Linear(embedding_dim, 2, bias=False), nn.Softmax(dim=-1)))
        self.embedding_dim = embedding_dim
 
        self.reset_parameters()
        self.alpha.data = th.zeros(self.order)
        self.alpha.data[0] = th.tensor(1.0)
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

    def pad_data(self, data):
        seq_len = [s.size(0) for s in data] 
        data = pad_sequence(data, batch_first=True)    
        return data, seq_len

    def forward(self, mg, item_ids, release_times, seq_sizes):
        # print(mg, seq_sizes, release_times)

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
       
        h = feats
        for idx, layer in enumerate(self.layers):
            h = layer(mg, h)

        last_nodes = []
        for i in range(self.order):
            if self.norm:
                h['s'+str(i+1)] = nn.functional.normalize(h['s'+str(i+1)], dim=-1)
            last_nodes.append(mg.filter_nodes(lambda nodes: nodes.data['last'] == 1, ntype='s'+str(i+1)))
            
        feat = h
        sr_g = self.readout(mg, feat, last_nodes)  

        sr_l = th.cat([feat['s'+str(i+1)][last_nodes[i]].unsqueeze(1) for i in range(self.order)], dim=1)
        
        # sr shape: B * order * embedding_dim
        # print(f"sr.shape: {sr.shape}")
        sr = 0.4*sr_l + 0.6*sr_g

        # time attention representation
        item_feat = self.embeddings(item_ids)
        item_feat = self.feat_drop(item_feat)
        if self.norm:
            item_feat = nn.functional.normalize(item_feat, dim=-1)

        time_feat = self.time_embeddings(release_times).view(-1, self.time_embedding_dim*4)
        time_feat = self.feat_drop(time_feat)

        sr_time = self.time_readout(time_feat, item_feat, seq_sizes)
        sr_time = sr_time.unsqueeze(1).repeat(1, self.order, 1)   
        # print(f"sr_time.shape: {sr_time.shape}")

        sr = th.cat([sr, sr_time], dim=-1)
        if self.norm:
            sr = nn.functional.normalize(sr, dim=-1)
        
        target_time_feat = self.time_embeddings(self.time_indices)\
            .view(-1, self.time_embedding_dim*4)

        target = self.embeddings(self.indices)

        target = th.cat(
            [target, target_time_feat],
            dim=-1
        )
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
            if self.training:
                if self.order == 1:
                    phi = phi.squeeze(1)
                    score = (th.cat((score.unsqueeze(1), score_ex.unsqueeze(1)), dim=1) * phi).sum(1)
                else:
                    score = (th.cat((score.unsqueeze(2), score_ex.unsqueeze(2)), dim=2) * phi).sum(2)
            else:
                score = score_ex
        else:
            logits = sr.squeeze() @ target.t()
            score  = th.softmax(12 * logits, dim=-1)
        
        if self.order > 1 and self.fusion:
            alpha = th.softmax(self.alpha.unsqueeze(0), dim=-1).view(1, self.alpha.size(0), 1)
            g = th.ones(score.size(0), score.size(1), 1).to(self.device)
            g = alpha.repeat(score.size(0), 1, 1)
            score = (score * g).sum(1)
        elif self.order > 1:
            score = score[:, 0]
            
        # print(score.shape)

        # score = nn.functional.log_softmax(score, dim=1)    
        score = th.log(score)
        
        return score
        
        
