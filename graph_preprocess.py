import json
from datasets import load_dataset
from tqdm import tqdm
import pickle
import argparse

def make_total_graph(dataset):
    # dataset = 'cwq'

    train_dataset = load_dataset(f"rmanluo/RoG-{dataset}", split='train')
    valid_dataset = load_dataset(f"rmanluo/RoG-{dataset}", split='validation')
    test_dataset = load_dataset(f"rmanluo/RoG-{dataset}", split='test')

    temp_set = set()
    for item in train_dataset:
        for t in item['graph']:
            temp_set.add(tuple(t))

    for item in valid_dataset:
        for t in item['graph']:
            temp_set.add(tuple(t))

    for item in test_dataset:
        for t in item['graph']:
            temp_set.add(tuple(t))

    temp_list = list(temp_set)
    final_list = []
    for t in temp_list:
        final_list.append(list(t))

    with open(f'/data/{dataset}/total_graph_{dataset}.jsonl', 'w', encoding='utf-8') as file:
        json.dump(final_list, file)
    
    triple2id = {trip : i for i, trip in enumerate(temp_list)}

    with open(f'/data/{dataset}/{dataset}{dataset}_triple2id.pkl', 'wb') as f:
        pickle.dump(triple2id,f)



def make_topic2graph(dataset):
    train_dataset = load_dataset(f"rmanluo/RoG-{dataset}", split='train')
    valid_dataset = load_dataset(f"rmanluo/RoG-{dataset}", split='validation')
    test_dataset = load_dataset(f"rmanluo/RoG-{dataset}", split='test')

    topic2graph = dict()

    for item in tqdm(train_dataset):
        q_list = item['q_entity']
        subgraph = item['graph']
        temp_graph = set()
        for triple in subgraph:
            temp_graph.add(tuple(triple))

        for entity in q_list:
            if entity not in topic2graph:
                topic2graph[entity] = set()
            topic2graph[entity] |= temp_graph

    for item in tqdm(valid_dataset):
        q_list = item['q_entity']
        subgraph = item['graph']
        temp_graph = set()
        for triple in subgraph:
            temp_graph.add(tuple(triple))

        for entity in q_list:
            if entity not in topic2graph:
                topic2graph[entity] = set()
            topic2graph[entity] |= temp_graph

    for item in tqdm(test_dataset):
        q_list = item['q_entity']
        subgraph = item['graph']
        temp_graph = set()
        for triple in subgraph:
            temp_graph.add(tuple(triple))

        for entity in q_list:
            if entity not in topic2graph:
                topic2graph[entity] = set()
            topic2graph[entity] |= temp_graph

    topic2graph = {k : list(v) for k, v in topic2graph.items()}

    with open(f'/data/{dataset}/{dataset}_triple2id.pkl', 'rb') as f:
        triple2id = pickle.load(f)

    new_graph = dict()
    for k, v in topic2graph.items():
        if k not in new_graph:
            new_graph[k] = list()
        new_graph[k] += [triple2id[triplet] for triplet in v]
        
    with open(f'/data/{dataset}/{dataset}_topic_graph.pkl', mode='wb') as f:
        pickle.dump(new_graph, f)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='webqsp')
args, unknown = parser.parse_known_args()

make_total_graph(args.dataset)
make_topic2graph(args.dataset)
