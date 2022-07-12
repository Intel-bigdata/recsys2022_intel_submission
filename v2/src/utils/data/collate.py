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
from collections import Counter
import numpy as np
import torch as th
import dgl
from random import sample


def label_last_ccs(g, last_nid):
    for i in range(len(last_nid)):
        is_last = th.zeros(g.num_nodes('s'+str(i+1)), dtype=th.int32)
        is_last[last_nid[i]] = 1
        g.nodes['s'+str(i+1)].data['last'] = is_last
    return g


def sample_list_with_ratio(src_dst_list, p=0):
    src_dst_list = list(src_dst_list)
    if not src_dst_list:
        return []
    sample_cnt = int(len(src_dst_list) * (1-p))
    sample_cnt = max(1, sample_cnt)
    src_dst_list = sample(src_dst_list, sample_cnt)
    return src_dst_list


def get_src_dst_from_list(src_dst_list, p=0):
    src_dst_list = sample_list_with_ratio(src_dst_list, p)
    counter = Counter(src_dst_list)
    edges = counter.keys()

    if len(edges) > 0:
        src, dst = zip(*edges)
    else:
        src, dst = [], []
    # (th.tensor(src, dtype=th.long), th.tensor(dst, dtype=th.long))
    return (src, dst)



def seq_to_ccs_graph(seq, order, feat_items_dict, cat_feats_dict, most_freq_feat, 
    item_cats_dict, cat_items_dict, edge_drop_ratio):
    
    class Combine:
        def __init__(self):
            self.dict = {}
        
        def __call__(self, *input):
            return self.forward(*input)   
        
        def forward(self, i, order):
            if str(i) not in self.dict:
                self.dict[str(i)] = {}
            if order not in self.dict[str(i)]:
                self.dict[str(i)][order] = com(i, order)
            return self.dict[str(i)][order]

    def com(i, order):
        item = str(seq[i:i+order])

        return item  

    order1 = order
    order = min(order, len(seq))
    feat_order = order1 + 1

    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}

    feats = np.unique(list(feat_items_dict.keys()))
    fiid2fnid = {iid: i for i, iid in enumerate(feats)}
    
    seq_nid = [iid2nid[iid] for iid in seq]
    last_k = [iid2nid[seq[-1]]]

    combine = Combine()  

    combine_seqs = []     
    item_dicts = [iid2nid]
    for i in range(1, order1):
        combine_seq = []
        item_dict = {}
        cnt = 0
        for j in range(len(seq_nid)-i):
            item = combine(j, i+1)
            if item not in item_dict:
                item_dict[item] = cnt
                cnt += 1
                combine_seq.append([seq[idx] for idx in range(j, j+i+1)])
    
        if len(item_dict) > 0:
            last_k.append(item_dict[item])
        else:
            last_k.append(0)
        combine_seqs.append(combine_seq)
                
        item_dicts.append(item_dict)
    
    graph_data = {}
    
    for k in range(1, order):
        intra_src_dst_list = [(item_dicts[k][combine(i, k+1)], item_dicts[k][combine(i+1, k+1)]) for i in range(len(seq)-k-1)]

        graph_data[('s'+str(k+1), 'intra'+str(k+1), 's'+str(k+1))] = get_src_dst_from_list(intra_src_dst_list, edge_drop_ratio)
        
    for k in range(1, order): 
        inter_src_dst_list = [(seq_nid[i], item_dicts[k][combine(i+1, k+1)]) for i in range(len(seq)-k-1)]
        graph_data[('s1', 'inter', 's'+str(k+1))] = get_src_dst_from_list(inter_src_dst_list, edge_drop_ratio)
       
        inter_src_dst_list = [(item_dicts[k][combine(i, k+1)], seq_nid[i+k+1]) for i in range(len(seq)-k-1)]
        graph_data[('s'+str(k+1), 'inter', 's1')] = get_src_dst_from_list(inter_src_dst_list, edge_drop_ratio)
    
    if order < order1:
        for i in range(order, order1):
            graph_data[('s'+str(i+1), 'intra'+str(i+1), 's'+str(i+1))]=(th.LongTensor([]), th.LongTensor([]))
            graph_data[('s'+str(i+1), 'inter', 's1')]=(th.LongTensor([]), th.LongTensor([]))
            graph_data[('s1', 'inter', 's'+str(i+1))]=(th.LongTensor([]), th.LongTensor([]))
    
    # add feat -> feat edge
    feat_src_feat_dst_list = []
    for _, feat_seq in cat_feats_dict.items():
        feat_seq_nid = [fiid2fnid[iid] for iid in feat_seq]
        feat_src_feat_dst_list += [(feat_seq_nid[i], feat_seq_nid[i+1]) for i in range(len(feat_seq)-1)]

    graph_data[(f's{feat_order}', f'intra{feat_order}', f's{feat_order}')] = get_src_dst_from_list(feat_src_feat_dst_list, edge_drop_ratio)
    
    # add item <->feat edge
    item_src, feat_dst = [], []
    for feat in feat_items_dict:
        item_src += list(map(lambda x: iid2nid[x], feat_items_dict[feat]))
        feat_dst += [ fiid2fnid[feat] ] * len(feat_items_dict[feat])
    graph_data[('s1', 'inter', f's{feat_order}')] = get_src_dst_from_list(zip(item_src, feat_dst), edge_drop_ratio)
    graph_data[(f's{feat_order}', 'inter', 's1')] = get_src_dst_from_list(zip(feat_dst, item_src), edge_drop_ratio)

    # add hop edge: item -> item
    # intra_src_dst_list = [(seq_nid[i], seq_nid[i+1]) for i in range(len(seq)-1)]
    intra_src_dst_list = []
    for _, items_seq in cat_items_dict.items():
        items_seq_nid = [iid2nid[iid] for iid in items_seq]
        intra_src_dst_list += [(items_seq_nid[i], items_seq_nid[i+1]) for i in range(len(items_seq)-1)]
    graph_data[('s1', 'intra1', 's1')] = get_src_dst_from_list(intra_src_dst_list)


    g = dgl.heterograph(graph_data)
    if g.num_nodes('s1') < len(items):
        g.add_nodes(len(items)-g.num_nodes('s1'), ntype='s1')
    g.nodes['s1'].data['iid'] = th.from_numpy(items)
    
    if order < order1:
        for i in range(order, order1):
            if 's'+str(i+1) not in g.ntypes or g.num_nodes('s'+str(i+1)) == 0:
                g.add_nodes(1, ntype='s'+str(i+1))
                g.nodes['s'+str(i+1)].data['iid'] = th.ones(1, i+1).long() * g.nodes['s1'].data['iid'][0]
                # print(g.nodes['s'+str(i+1)].data)
    for i in range(1, order):
        if g.num_nodes('s'+str(i+1)) == 0:
            g.add_nodes(1, ntype='s'+str(i+1))
        if g.num_nodes(f's{i+1}') < len(combine_seqs[i-1]):
            g.add_nodes(len(combine_seqs[i-1])-g.num_nodes(f's{i+1}'), ntype=f's{i+1}')
        g.nodes['s'+str(i+1)].data['iid'] = th.from_numpy(np.array(combine_seqs[i-1]))

    # add feat iid and last node
    if g.num_nodes(f's{feat_order}') < len(feats):
        g.add_nodes(len(feats)-g.num_nodes(f's{feat_order}'), ntype=f's{feat_order}')
    g.nodes[f's{feat_order}'].data['iid'] = th.from_numpy(feats)
    last_k.append(fiid2fnid[most_freq_feat])

    label_last_ccs(g, last_k)

    return g


def collate_fn_factory_ccs(seq_to_graph_fns, order=1, edge_drop_ratio=0):
    def collate_fn(samples):
        seqs, labels, feat_items_dicts, cat_feats_dicts, \
            most_freq_feats, item_cats_dicts, cat_items_dicts, release_time_lists = zip(*samples)

        inputs = []
        graphs = []

        cnt = 0
        for seq_to_graph in seq_to_graph_fns:
            batch = list(map(
                seq_to_graph, 
                seqs, [order for _ in range(len(seqs))],
                feat_items_dicts, cat_feats_dicts, most_freq_feats, item_cats_dicts, cat_items_dicts,
                [edge_drop_ratio] * len(seqs)
            ))
            if cnt == 0:
                for idx, bh in enumerate(batch):
                    graph = bh
                    graphs.append(graph)
                bg = dgl.batch(graphs)
                cnt = 1
            else:
                bg = dgl.batch(batch)
            inputs.append(bg)

        item_ids = th.IntTensor([y for x in seqs for y in x])
        release_times = th.IntTensor([y for x in release_time_lists for y in x])
        seq_sizes = th.IntTensor([len(seq) for seq in seqs])
        inputs += [item_ids, release_times, seq_sizes]

        labels = th.LongTensor(labels)

        return inputs, labels

    return collate_fn


if __name__ == '__main__':
    print(sample_list_with_ratio([1,2,3,4,5], p=0.2))

    print('\n')
    seq = [4, 1, 3, 2, 2]
    feat_items_dict = {5: set([3, 1]), 6: set([4, 2])}
    cat_feats_dict = {11: [5], 12:[6]}
    item_cats_dict = {1:'11', 2:'12', 3:'11', 4:'12'}
    cat_items_dict = {'11': [1, 3], '12': [4, 2]}
    most_freq_feat = 6
    g1 = seq_to_ccs_graph(
        seq, 3, 
        feat_items_dict, cat_feats_dict, most_freq_feat, item_cats_dict, cat_items_dict,
        0.8
    )
    print(g1)
    # print(g1.nodes['s2'])
    print(g1.nodes['s1'].data)
    print(g1.nodes['s2'].data)
    print(g1.nodes['s3'].data)
    print(g1.nodes['s4'].data)
    print(g1.number_of_nodes())
    print(g1.edges(etype=('s1', 'intra1', 's1'), form='all'))
    print(g1.edges(etype=('s4', 'intra4', 's4'), form='all'))
    print(g1.edges(etype=('s1', 'inter', 's4'), form='all'))
    print(g1.edges(etype=('s4', 'inter', 's1'), form='all'))


    # print('\n')
    # seq2 = [3, 4]
    # feat_items_dict2 = {5: set([3, 4])}
    # cat_feats_dict2 = {11: [5]}
    # most_freq_feat2 = 5
    # collate_fn = collate_fn_factory_ccs((seq_to_ccs_graph,), order=3)
    # seqs = [
    #     [seq, 1, feat_items_dict, cat_feats_dict, most_freq_feat, []], 
    #     [seq2, 2, feat_items_dict2, cat_feats_dict2, most_freq_feat2, []]
    # ]
    # mg = collate_fn(seqs)[0][0]
    # # # print(f'label: {collate_fn(seqs)[1]}')
    # print(f'graph: {mg}')
    # print(mg.nodes['s1'].data['iid'])
    # print(mg.nodes['s2'].data['iid'])
    # # print(mg.nodes['s3'].data['iid'])
    # print(mg.batch_num_nodes('s1'))
    # print(mg.batch_num_nodes('s2'))
    # # print(mg.batch_num_nodes('s3'))

