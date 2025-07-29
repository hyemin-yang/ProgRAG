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

    # Query는 이미 1차원 시퀀스이므로, 각 샘플별로 padding 처리 후 배치로 만듭니다.
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

    # 배치 크기가 1이라고 가정하고, 한 샘플 내의 여러 path 시퀀스들을 처리하는 헬퍼 함수.
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
        'query_id': batch_data[0]['id'], #배치1 가정
        'query_hop': batch_data[0]['query_hop'], #배치1 가정
        'query_token_ids': query_token_ids,           # [1, seq_len]
        'query_mask': query_mask,                     # [1, seq_len]
        'query_token_type_ids': query_token_type_ids, # [1, seq_len]
        'neg_token_ids': neg_token_ids,               # [1, num_neg_paths, seq_len]
        'neg_mask': neg_mask,                         # [1, num_neg_paths, seq_len]
        'neg_token_type_ids': neg_token_type_ids,     # [1, num_neg_paths, seq_len]
        'pos_token_ids': pos_token_ids,               # [1, num_pos_paths, seq_len]
        'pos_mask': pos_mask,                         # [1, num_pos_paths, seq_len]
        'pos_token_type_ids': pos_token_type_ids,      # [1, num_pos_paths, seq_len]
        'cand_triple_ids': batch_data[0]['pos_triple_ids'] + batch_data[0]['neg_triple_ids']    #배치1가정
    }
    return batch_dict


# def collate(batch_data: List[dict]) -> dict:
#     pad_token_id = tokenizer.pad_token_id

#     # Query는 이미 1차원 시퀀스이므로, 각 샘플별로 padding 처리 후 배치로 만듭니다.
#     query_token_ids, query_mask = to_indices_and_mask(
#         [torch.LongTensor(ex['query_token_ids']) for ex in batch_data],
#         pad_token_id=pad_token_id,
#         need_mask=True
#     )
#     query_token_type_ids = to_indices_and_mask(
#         [torch.LongTensor(ex['query_token_type_ids']) for ex in batch_data],
#         pad_token_id=pad_token_id,
#         need_mask=False
#     )

#     # 배치 크기가 1이라고 가정하고, 한 샘플 내의 여러 path 시퀀스들을 처리하는 헬퍼 함수.
#     def process_field(field_name):
#         sample_seqs = batch_data[0][field_name] 
#         if len(sample_seqs) == 0:
#             seq_tensors = [torch.LongTensor([pad_token_id])]
#         else:
#             seq_tensors = [torch.LongTensor(seq) for seq in sample_seqs]
#         indices, mask = to_indices_and_mask(seq_tensors, pad_token_id=pad_token_id, need_mask=True)
#         return indices.unsqueeze(0), mask.unsqueeze(0)

#     triple_token_ids, triple_mask = process_field('triple_token_ids')
#     triple_token_type_ids, _ = process_field('triple_token_type_ids')
#     labels = torch.tensor(batch_data[0]['labels'])
#     batch_dict = {
#         'query_token_ids': query_token_ids,           # [1, seq_len]
#         'query_mask': query_mask,                     # [1, seq_len]
#         'query_token_type_ids': query_token_type_ids, # [1, seq_len]
#         'triple_token_ids': triple_token_ids,               # [1, num_neg_paths, seq_len]
#         'triple_mask': triple_mask,                         # [1, num_neg_paths, seq_len]
#         'triple_token_type_ids': triple_token_type_ids,     # [1, num_neg_paths, seq_len]
#         'labels': labels #배치1가정
#     }
#     return batch_dict


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

def split_path(paths, triple2id): #broken = [path1, path2,...] (path1 = [(t1),(t2),...]), gold_ids, golden_ent_ids(중간 노드랑, 끝 노드만)
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


#내가 원하는 output: sample[0]의 head와 이웃인 triple들. golden은 아닌. 
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
    
    # 이웃에서 찾은 triple이 모두 golden_ids에 속해서 cand_triples가 비어있다면,
    # 전체 그래프의 edge들에서 golden_ids에 없는 triple들을 샘플링.
    if not cand_triples:
        extra = set(id2triple.keys()) - golden_ids
        random_samples = random.sample(extra, min(len(extra), max_num_neg))  

        for triple_id in random_samples:
            triple = id2triple[triple_id]
            cand_triples.append(triple)
            cand_triple_ids.append(triple_id)

        # 4. golden 트리플 정보
        goldens = [id2triple[idx] for idx in golden_ids]
        # breakpoint()
        # for u, v, data in graph.edges(data=True):
        #     triple_id = data.get('triple')
        #     triple = id2triple[triple_id]
        #     if triple_id not in golden_ids: #and any(ent not in golden_ents for ent in (triple[0], triple[-1])):
        #         cand_triples.append(triple)
        #         cand_triple_ids.append(triple_id)
        #         if len(cand_triples) >= (max_num_neg-len(golden_ids)):
        #             break
        # goldens = [id2triple[idx] for idx in golden_ids]
    
    # 추출된 후보가 max_num_neg를 초과하면, 최대 max_num_neg개만 사용합니다.
    if len(cand_triples) > (max_num_neg-len(goldens)):
        cand_triples = cand_triples[:max_num_neg-len(goldens)]
        cand_triple_ids = cand_triple_ids[:max_num_neg-len(goldens)]

    return cand_triples, cand_triple_ids, goldens


# def extract_cand_path(graph, id2triple, golden_triples, max_num_neg, golden_ids, golden_ents, id_subgraph):
#     cand_triples = list()
#     cand_triple_ids = list()

#     for path in golden_triples:
#         for start in path:
#             neighbors = graph[start[0]]
#             for tail in neighbors:
#                 if tail not in golden_ents:
#                     triples = find_triple(graph, start[0], tail, id2triple)
#                     for k, v in triples.items():
#                         if k not in golden_ids:
#                             cand_triples.append(v)
#                             cand_triple_ids.append(k)
        
#     # 이웃에서 찾은 triple이 모두 golden_ids에 속해서 cand_triples가 비어있다면,
#     # 전체 그래프의 edge들에서 golden_ids에 없는 triple들을 샘플링.
#     if not cand_triples:
#         extra =  id_subgraph - golden_ids
#         random_samples = random.sample(extra, min(len(extra), max_num_neg))  
 
#         for triple_id in random_samples:
#             triple = id2triple[triple_id]
#             cand_triples.append(triple)
#             cand_triple_ids.append(triple_id)


#     if len(cand_triples) > max_num_neg:
#         cand_triples = cand_triples[:max_num_neg]
#         cand_triple_ids = cand_triple_ids[:max_num_neg]
 
#     return cand_triples, cand_triple_ids


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

# def new_load_data(path: str, graph_path, triple2id: str, max_num_neg: int, max_num_pos:int, tokenizer):
#     #data: query, golden_path
#     data = list()
#     with open(path, 'r') as f:
#         for line in f:
#             data.append(json.loads(line))
#     print('Load {} examples from {}'.format(len(data), path))
    
#     with open(triple2id, 'rb') as f:
#         triple2id = pickle.load(f)
#     id2triple = dict()
#     for k,v in triple2id.items():
#         id2triple[v] = k 
#     ddd = []
#     with open(graph_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             ddd.append(json.loads(line))        
#     triples = ddd[0]
#     graph = get_undirected_graph(triples, triple2id)       
#     print('Load {} triples from {}'.format(len(triples), graph_path))
    
#     cnt = 0
#     examples = []
#     for i in tqdm(range(len(data))):
#         ex = data[i]
#         if len(ex["golden_path"]) == 0:
#             cnt+=1
#             continue

#         query = ex["question"]
#         golden_triples, golden_ids, golden_ents = split_path(ex["golden_path"], triple2id)
#         visited_1hop = set()
#         visited_2hop = set()
#         for sample in golden_triples:
#             query_hop = len(sample)
#             #assume query_hop is 1 or 2
#             if query_hop == 1 and sample[0] not in visited_1hop:
#                 cand_triples, cand_triple_ids, pos_triples = extract_cand_path(graph, id2triple, sample[0], max_num_neg, max_num_pos, golden_ids, golden_ents)
#                 visited_1hop = visited_1hop.union(pos_triples)
#                 cand_triples_str = [' '.join(t) for t in cand_triples]
#                 pos_triples_str = [' '.join(t) for t in pos_triples]
#                 pos_triple_ids = [triple2id[v] for v in pos_triples]
#                 encoded = encode_input(query, cand_triples_str, cand_triple_ids, pos_triples_str, pos_triple_ids, tokenizer)
#                 encoded['id'] = ex['id']
#                 encoded['query_hop'] = query_hop
#                 examples.append(encoded)
            
#             elif query_hop == 2:
#                 #check id entity
#                 inter_ent = sample[1][0]
#                 if len(inter_ent)>1 and inter_ent[1]=='.' and sample[1] not in visited_2hop:
#                     augmented_query = query
#                     cand_triples, cand_triple_ids, pos_triples = extract_cand_path(graph, id2triple, sample[1], max_num_neg, max_num_pos, golden_ids, golden_ents)
#                     visited_2hop = visited_2hop.union(pos_triples)
#                     cand_triples_str = []
#                     pos_triples_str = []
#                     for t in cand_triples:
#                         cand_triples_str.append(sample[0][0] + ' '+ t[1]+ ' '+ t[2])  #change id_entity to meaningful_entity (topic entity)  
#                     for t in pos_triples:
#                         pos_triples_str.append(sample[0][0] + ' '+ t[1]+ ' '+ t[2])
#                     pos_triple_ids = [triple2id[v] for v in pos_triples]
#                     encoded = encode_input(augmented_query, cand_triples_str, cand_triple_ids, pos_triples_str, pos_triple_ids, tokenizer)
#                     encoded['id'] = ex['id']
#                     encoded['query_hop'] = query_hop
#                     examples.append(encoded)
                    
#                 else:
#                     if sample[0] not in visited_1hop:
#                         #make 2 sample
#                         cand_triples1, cand_triple_ids1, pos_triples1= extract_cand_path(graph, id2triple, sample[0], max_num_neg, max_num_pos, golden_ids, golden_ents) 
#                         visited_1hop = visited_1hop.union(pos_triples1)
                
#                         cand_triples_str = [' '.join(t) for t in cand_triples1]
#                         pos_triples_str = [' '.join(t) for t in pos_triples1]
#                         pos_triple_ids1 = [triple2id[v] for v in pos_triples1]
#                         encoded = encode_input(query, cand_triples_str, cand_triple_ids1, pos_triples_str, pos_triple_ids1 , tokenizer)
#                         encoded['id'] = ex['id']
#                         encoded['query_hop'] = query_hop
#                         examples.append(encoded)
                    
#                     if sample[1] not in visited_2hop:
#                         augmented_query = query + ' ' + ' '.join(sample[0]) #query + 1hop path
#                         cand_triples, cand_triple_ids, pos_triples = extract_cand_path(graph, id2triple, sample[1], max_num_neg, max_num_pos, golden_ids, golden_ents)           
#                         visited_2hop =  visited_2hop.union(pos_triples)
#                         cand_triples_str = [' '.join(t) for t in cand_triples]
#                         pos_triples_str = [' '.join(t) for t in pos_triples]
#                         pos_triple_ids = [triple2id[v] for v in pos_triples]
#                         encoded = encode_input(augmented_query , cand_triples_str, cand_triple_ids, pos_triples_str, pos_triple_ids, tokenizer)
#                         encoded['id'] = ex['id']
#                         encoded['query_hop'] = query_hop
#                         examples.append(encoded)
                   
        
#     print(f'Skip {cnt} samples')
#     print(f'Load {len(examples)} samples')
#     print('Loading data has been finished')
    
#     return examples, triple2id, id2triple


def new_load_data(path: str, graph_path, triple2id_path: str, max_num_neg: int, max_num_pos: int, tokenizer, max_query_hop: int = 4):
    import json, pickle
    from tqdm import tqdm

    # 데이터 로딩: query와 golden_path들을 포함한 예시들
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print('Load {} examples from {}'.format(len(data), path))
    
    # triple2id 및 id2triple 로딩
    with open(triple2id_path, 'rb') as f:
        triple2id = pickle.load(f)
    # with open(triple2id_path, 'r', encoding='utf-8') as f:
    #     triple2id  = json.load(f)
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
    
    # 최대 hop 수(max_query_hop) 이상인 샘플은 처리하지 않도록 하거나, 필요시 max_query_hop까지만 사용
    #for i in tqdm(range(10)):
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
        # 각 hop마다 이미 처리한 triple을 기록하기 위한 dict (key: hop index, value: set)
        visited = {hop: set() for hop in range(max_query_hop)}
        #print(len(golden_triples))
        for sample in golden_triples:
            #breakpoint()
            query_hop = len(sample)
            # 너무 깊은 경로는 건너뛰거나, max_query_hop까지만 사용
            if query_hop > max_query_hop:
                continue
            # 각 hop(즉, 각 triple)에 대해 처리: 이전 hop들의 정보를 query에 붙여서 augmented query를 구성
            for hop_idx in range(query_hop):
                # 만약 이 hop의 triple이 이미 처리되었으면 건너뜁니다.
                if sample[hop_idx] in visited[hop_idx]:
                    continue
                
                # augmented query 구성:
                # - hop 0인 경우: 원래 query 그대로 사용
                # - hop > 0인 경우: 만약 현재 triple의 head가 의미 없는 id라면 원 query만 사용하고,
                #   그렇지 않으면 이전 hop들의 triple들을 문자열로 이어붙여서 query를 보강합니다.
                if hop_idx == 0:
                    aug_query = query
                else:
                    if len(sample[hop_idx][0]) > 1 and sample[hop_idx][0][1] == '.':
                        aug_query = query
                    else:
                        # 이전 hop들의 triple을 "head relation tail" 형태로 이어붙임
                        prev_info = ' '.join([' '.join(tr) for tr in sample[:hop_idx]])
                        aug_query = query + ' ' + prev_info
                        #breakpoint()
                #s = time.time()
                # 현재 hop의 triple(sample[hop_idx])을 시작점으로 후보 경로 추출
                cand_triples, cand_triple_ids, pos_triples = extract_cand_path(
                    graph, id2triple, sample[hop_idx], max_num_neg, max_num_pos, golden_ids, golden_ents
                )
                #e = time.time()
                #print('extract path time: ',datetime.timedelta(seconds=e-s))
                # 해당 hop에 대해 추출한 긍정 triple들을 visited에 추가
                visited[hop_idx].update(pos_triples)
                
                if len(cand_triples) == 0:
                    error +=1
                    continue
                # 후보와 긍정 triple의 string 표현 생성:
                # 만약 현재 triple의 head가 의미 없는 id라면, sample[0][0] (첫 triple의 head)를 사용해 대체합니다.
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
                encoded['query_hop'] = query_hop  # 전체 path 길이를 저장
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
                    encoded['query_hop'] = query_hop  # 전체 path 길이를 저장
                    examples.append(encoded)
    print(f'Skip {cnt} samples')
    print(f'Skip {error} samples : do not have negative')
    print(f'Load {len(examples)} samples')
    print('Loading data has been finished')
    
    return examples, triple2id, id2triple


def allnew_load_data(query_path, graph_path, triple2id_path: str,max_num_neg: int, max_num_pos: int, tokenizer, max_query_hop: int = 4):
    # 데이터 로딩: query와 golden_path들을 포함한 예시들
    data = []
    with open(query_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print('Load {} examples from {}'.format(len(data), query_path))
    
    # triple2id 및 id2triple 로딩
    with open(triple2id_path, 'rb') as f:
        triple2id = pickle.load(f)
    # with open(triple2id_path, 'r', encoding='utf-8') as f:
    #     triple2id  = json.load(f)
    id2triple = {v: k for k, v in triple2id.items()}
    
    
    
    # 그래프 로딩: graph_path 파일은 jsonl 형식이며, 첫 줄에 triples 리스트가 있다고 가정
    ddd = []
    with open(graph_path, 'r', encoding='utf-8') as f:
        for line in f:
            ddd.append(json.loads(line))
    triples = ddd[0]
 
    graph = get_undirected_graph(triples, triple2id)
    print('Load {} triples from {}'.format(len(triples), graph_path))
    
    
    #data 처리 너무 오래걸리 시 사용하기 (cpu cost 큼)
    # triple_str = [' '.join(list(triple)) for triple in list(triple2id.keys())]
    # encoded_triples = tokenizer(triple_str, add_special_tokens=True, max_length=200,
    #                                     return_token_type_ids=True, truncation=True)
        
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

