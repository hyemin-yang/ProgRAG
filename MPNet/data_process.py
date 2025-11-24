import pickle
import json
import torch
#from .config import args
import random
import networkx as nx
from typing import Optional, List
import torch.utils.data.dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import time
import datetime

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        return self.examples[index]


tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")


def collate(batch_data: List[dict]) -> dict:
    pad_token_id = tokenizer.pad_token_id

    query_token_ids, query_mask = to_indices_and_mask(
        [torch.LongTensor(ex['query_token_ids']) for ex in batch_data],
        pad_token_id=pad_token_id,
        need_mask=True
    )
    query_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['query_token_type_ids']) for ex in batch_data],
        pad_token_id=pad_token_id,
        need_mask=False
    )

    def process_field(field_name):
        sample_seqs = batch_data[0][field_name] 
        if len(sample_seqs) == 0:
            seq_tensors = [torch.LongTensor([pad_token_id])]
        else:
            seq_tensors = [torch.LongTensor(seq) for seq in sample_seqs]
        indices, mask = to_indices_and_mask(seq_tensors, pad_token_id=pad_token_id, need_mask=True)
        return indices.unsqueeze(0), mask.unsqueeze(0)

    neg_token_ids, neg_mask = process_field('neg_token_ids')
    neg_token_type_ids, _ = process_field('neg_token_type_ids')
    pos_token_ids, pos_mask = process_field('pos_token_ids')
    pos_token_type_ids, _ = process_field('pos_token_type_ids')

    batch_dict = {
        'query_id': batch_data[0]['id'], 
        'query_hop': batch_data[0]['query_hop'],
        'query_token_ids': query_token_ids,           # [1, seq_len]
        'query_mask': query_mask,                     # [1, seq_len]
        'query_token_type_ids': query_token_type_ids, # [1, seq_len]
        'neg_token_ids': neg_token_ids,               # [1, num_neg_paths, seq_len]
        'neg_mask': neg_mask,                         # [1, num_neg_paths, seq_len]
        'neg_token_type_ids': neg_token_type_ids,     # [1, num_neg_paths, seq_len]
        'pos_token_ids': pos_token_ids,               # [1, num_pos_paths, seq_len]
        'pos_mask': pos_mask,                         # [1, num_pos_paths, seq_len]
        'pos_token_type_ids': pos_token_type_ids,      # [1, num_pos_paths, seq_len]
        'cand_triple_ids': batch_data[0]['pos_triple_ids'] + batch_data[0]['neg_triple_ids']   
    }
    return batch_dict

def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
    
def get_undirected_graph(triples, triple2id):
    G = nx.MultiGraph()
    for idx, triple in enumerate(triples):
        h, r, t = triple
        G.add_edge(h, t, r, triple = triple2id[(h,r,t)])
    return G

def get_undirected_graph_subgraph(triples, id2triple):
    G = nx.MultiGraph()
    for triple in triples:
        h, r, t = id2triple[triple]
        G.add_edge(h, t, r, triple = triple)
    return G


def find_triple(graph, e1, e2, id2triple):
    if graph.has_edge(e1, e2):
        edge_data = graph.get_edge_data(e1, e2)
        triples = {}
        for key in edge_data.keys():
            id = edge_data[key]['triple']
            triples[id] = id2triple[id]
        return triples
    return {}

def split_path(paths, triple2id): #broken = [path1, path2,...] (path1 = [(t1),(t2),...]), gold_ids, golden_ent_ids
    broken = list()
    golden_ids = set()
    golden_ents = set()  
    for p in paths:
        one = []
        for i in range(0,len(p)-2,2):
            triple = tuple(p[i:i+3])
            one.append(triple)
            if triple in triple2id:
                #one.append(triple)
                golden_ents.add(triple[-1]) 
                golden_ids.add(triple2id[triple])
            else:
                #find reverse triple
                triple = (p[i+2], p[i+1], p[i])
                #one.append(triple)
                golden_ents.add(triple[0]) 
                golden_ids.add(triple2id[triple])
        broken.append(one)
    return broken, golden_ids, golden_ents 


def extract_cand_path(graph, id2triple, start, max_num_neg, max_num_pos, golden_ids, golden_ents):
    neighbors = list(graph.neighbors(start[0]))
    cand_triples = list()
    cand_triple_ids = list()
    goldens = set()
    
    for tail in neighbors:
        if tail not in golden_ents:
            triples = find_triple(graph, start[0], tail, id2triple)
            for k, v in triples.items():
                if k not in golden_ids:
                    cand_triples.append(v)
                    cand_triple_ids.append(k)
                else:
                    goldens.add(v)
        else:
            triples = find_triple(graph, start[0], tail, id2triple)
            if len(goldens) <= max_num_pos:
                for k, v in triples.items():
                    goldens.add(v)
    

    if not cand_triples:
        extra = set(id2triple.keys()) - golden_ids
        random_samples = random.sample(extra, min(len(extra), max_num_neg))  

        for triple_id in random_samples:
            triple = id2triple[triple_id]
            cand_triples.append(triple)
            cand_triple_ids.append(triple_id)

  
    if len(cand_triples) > (max_num_neg-len(goldens)):
        cand_triples = cand_triples[:max_num_neg-len(goldens)]
        cand_triple_ids = cand_triple_ids[:max_num_neg-len(goldens)]

    return cand_triples, cand_triple_ids, goldens


def encode_input(query, cand_triples, cand_triple_ids, pos_triples, pos_triple_ids, tokenizer):
    encoded_query = tokenizer(query, add_special_tokens=True, max_length=200, 
                                        return_token_type_ids=True, truncation=True)
    encoded_negs = tokenizer(cand_triples, add_special_tokens=True, max_length=200, 
                            return_token_type_ids=True, truncation=True)
    encoded_pos = tokenizer(pos_triples, add_special_tokens=True, max_length=200, 
                            return_token_type_ids=True, truncation=True)
    
    encoded = {'query_token_ids': encoded_query['input_ids'],
            'query_token_type_ids': encoded_query['token_type_ids'],
            'neg_token_ids': encoded_negs['input_ids'],
            'neg_token_type_ids': encoded_negs['token_type_ids'],
            'pos_token_ids':encoded_pos['input_ids'],
            'pos_token_type_ids':encoded_pos['token_type_ids'],
            'neg_triple_ids':cand_triple_ids,
            'pos_triple_ids': pos_triple_ids
            }
    return encoded



def new_load_data(path: str, graph_path, triple2id_path: str, max_num_neg: int, max_num_pos: int, tokenizer, max_query_hop: int = 4):
    import json, pickle
    from tqdm import tqdm

    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print('Load {} examples from {}'.format(len(data), path))
    
    with open(triple2id_path, 'rb') as f:
        triple2id = pickle.load(f)

    id2triple = {v: k for k, v in triple2id.items()}
    
    ddd = []
    # Option 1: negative sampling in total graph 
    with open(graph_path, 'r', encoding='utf-8') as f:
        for line in f:
            ddd.append(json.loads(line))
    triples = ddd[0]
    graph = get_undirected_graph(triples, triple2id)
    print('Load {} triples from {}'.format(len(triples), graph_path))
    
    # Option 2: negative sampling in subgraph
    # with open(graph_path, 'rb') as f:
    #     id2subgraph = pickle.load(f)
    # print('Load {} subgraphs from {}'.format(len(id2subgraph), graph_path))
       
    cnt = 0
    error = 0
    examples = []
    
    for i in tqdm(range(len(data))):
        ex = data[i]
        
        # unblock below 4 lines for option 2. 
        # if len(ex["golden_path"]) == 0 or ex["id"] not in id2subgraph:
        #     cnt +=1
        #     continue
        # graph = get_undirected_graph_subgraph(id2subgraph[ex["id"]], id2triple)
        
        # unblock below 3 lines for option 1.
        if len(ex["golden_path"]) == 0 :
            cnt +=1
            continue
        
        query = ex["question"]
        golden_triples, golden_ids, golden_ents = split_path(ex["golden_path"], triple2id)
        visited = {hop: set() for hop in range(max_query_hop)}
        for sample in golden_triples:
            query_hop = len(sample)
            if query_hop > max_query_hop:
                continue
        
            for hop_idx in range(query_hop):
                if sample[hop_idx] in visited[hop_idx]:
                    continue
            
                if hop_idx == 0:
                    aug_query = query
                else:
                    if len(sample[hop_idx][0]) > 1 and sample[hop_idx][0][1] == '.':
                        aug_query = query
                    else:
                        prev_info = ' '.join([' '.join(tr) for tr in sample[:hop_idx]])
                        aug_query = query + ' ' + prev_info

                cand_triples, cand_triple_ids, pos_triples = extract_cand_path(
                    graph, id2triple, sample[hop_idx], max_num_neg, max_num_pos, golden_ids, golden_ents
                )
             
                visited[hop_idx].update(pos_triples)
                
                if len(cand_triples) == 0:
                    error +=1
                    continue
               
                if len(sample[hop_idx][0]) > 1 and sample[hop_idx][0][1] == '.':
                    cand_triples_str = [sample[0][0] + ' ' + t[1] + ' ' + t[2] for t in cand_triples]
                    pos_triples_str = [sample[0][0] + ' ' + t[1] + ' ' + t[2] for t in pos_triples]
                else:
                    cand_triples_str = [' '.join(t) for t in cand_triples]
                    pos_triples_str = [' '.join(t) for t in pos_triples]
                
                pos_triple_ids = [triple2id[v] for v in pos_triples]
                encoded = encode_input(
                    aug_query, cand_triples_str, cand_triple_ids, pos_triples_str, pos_triple_ids, tokenizer
                )
                encoded['id'] = ex['id']
                encoded['query_hop'] = query_hop
                examples.append(encoded)
                
                if hop_idx >= 1:
                    cand_triples_str = []
                    pos_triples_str = []
                    before_path = ''
                    for bt in sample[:hop_idx]:
                        if before_path:
                            before_path += ' '
                            
                        before_path += ' '.join(list(bt)[:2])
                    for t in cand_triples:
                        cand_triples_str.append(before_path + ' ' + ' '.join(t))
                    
                    pos_triples_str = [before_path + ' ' + ' '.join(sample[hop_idx])]
                    encoded = encode_input(
                    query, cand_triples_str, cand_triple_ids, pos_triples_str, pos_triple_ids, tokenizer
                )
                    encoded['id'] = ex['id']
                    encoded['query_hop'] = query_hop 
                    examples.append(encoded)
    print(f'Skip {cnt} samples')
    print(f'Skip {error} samples : do not have negative')
    print(f'Load {len(examples)} samples')
    print('Loading data has been finished')
    
    return examples, triple2id, id2triple


def allnew_load_data(query_path, graph_path, triple2id_path: str,max_num_neg: int, max_num_pos: int, tokenizer, max_query_hop: int = 4):
    data = []
    with open(query_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print('Load {} examples from {}'.format(len(data), query_path))
    
    with open(triple2id_path, 'rb') as f:
        triple2id = pickle.load(f)
    id2triple = {v: k for k, v in triple2id.items()}
    
    ddd = []
    with open(graph_path, 'r', encoding='utf-8') as f:
        for line in f:
            ddd.append(json.loads(line))
    triples = ddd[0]
 
    graph = get_undirected_graph(triples, triple2id)
    print('Load {} triples from {}'.format(len(triples), graph_path))
    
        
    cnt = 0
    error = 0
    examples = []
    
    #for i in tqdm(range(5)):
    for i in tqdm(range(len(data))):
        ex = data[i]
        if len(ex["golden_path"]) == 0:
            cnt += 1
            continue

        query = ex["question"]
        golden_triples, golden_ids, golden_ents = split_path(ex["golden_path"], triple2id)
        neg_triples, neg_triple_ids= extract_cand_path(graph, id2triple, golden_triples, max_num_neg, golden_ids, golden_ents)
        
        cand_triples = golden_triples + neg_triples
        cand_triple_ids = list(golden_ids) + neg_triple_ids
        labels = [1]*len(golden_ids) + [0]*len(neg_triple_ids)
        
        encoded_query = tokenizer(query, add_special_tokens=True, max_length=200, 
                                        return_token_type_ids=True, truncation=True)

        cand_token_ids = [encoded_triples['input_ids'][idx] for idx in cand_triple_ids]
        cand_token_type_ids = [encoded_triples['token_type_ids'][idx] for idx in cand_triple_ids]
        
       
        sample = {'query_token_ids': encoded_query['input_ids'],
            'query_token_type_ids': encoded_query['token_type_ids'],
            'triple_token_ids': cand_token_ids,
            'triple_token_type_ids': cand_token_type_ids,
            'labels' : labels
            }
        examples.append(sample)
            
        
    print(f'Skip {cnt} samples')
    #print(f'Skip {error} samples : do not have negative')
    print(f'Load {len(examples)} samples')
    print('Loading data has been finished')
    
    return examples, triple2id, id2triple

