from torch_scatter import scatter_add
import json
import pickle
import torch
from torch_geometric.data import Data
import datasets
from nbfvariadic import *

def get_emb(temp, dataset): 
    if dataset == 'webqsp':
        relemb_path = '/home/hyemin/shared_data/webqsp/relation.pth'
    else:
        relemb_path = '/home/hyemin/shared_data/cwq/relation.pth'
    rel_emb = torch.load(relemb_path)
    rel2id = temp['rel2id']
    rels = list(rel2id.keys())
    selected_tensors = [rel_emb[key] for key in rels if key in rel_emb]
    merged_tensor = torch.stack(selected_tensors, dim=0)

    return merged_tensor

def entities_to_mask(entities, num_nodes):
    mask = torch.zeros(num_nodes)
    mask[entities] = 1
    return mask

def build_relation_graph(graph):
    # expect the graph is already with inverse edges

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T,
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]],
        (num_rels, num_nodes),
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, torch.ones(Eh.shape[0], device=device), (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T,
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]],
        (num_rels, num_nodes),
    )
    Et = torch.sparse_coo_tensor(
        Et.T, torch.ones(Et.shape[0], device=device), (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat(
        [
            Ahh.indices().T,
            torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0),
        ],
        dim=1,
    )  # head to head
    tt_edges = torch.cat(
        [
            Att.indices().T,
            torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1),
        ],
        dim=1,
    )  # tail to tail
    ht_edges = torch.cat(
        [
            Aht.indices().T,
            torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2),
        ],
        dim=1,
    )  # head to tail
    th_edges = torch.cat(
        [
            Ath.indices().T,
            torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3),
        ],
        dim=1,
    )  # tail to head

    rel_graph = Data(
        edge_index=torch.cat(
            [
                hh_edges[:, [0, 1]].T,
                tt_edges[:, [0, 1]].T,
                ht_edges[:, [0, 1]].T,
                th_edges[:, [0, 1]].T,
            ],
            dim=1,
        ),
        edge_type=torch.cat(
            [hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0
        ),
        num_nodes=num_rels,
        num_relations=4,
    )

    graph.relation_graph = rel_graph
    return graph

def load_file(item, id2triple, inv_entity_vocab: dict, inv_rel_vocab: dict) -> dict:
        """Load a knowledge graph file and return the processed data."""
        subgraph = item
        # subgraph = list(item.values())
        # id2triple = dict()
        # for k,v in triple2id.items():
        #     id2triple[v] = k

        triplets = []  # Triples with inverse relations
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)
        
        # if len(item['subgraph']) != 0:
        #     for t in item['subgraph']:
        #         u, r, v = id2triple[t]
        #         if u not in inv_entity_vocab:
        #             inv_entity_vocab[u] = entity_cnt
        #             entity_cnt += 1
        #         if v not in inv_entity_vocab:
        #             inv_entity_vocab[v] = entity_cnt
        #             entity_cnt += 1
        #         if r not in inv_rel_vocab:
        #             inv_rel_vocab[r] = rel_cnt
        #             rel_cnt += 1
        #         u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]
        #         triplets.append((u, v, r))

        # if len(item['subgraph']) != 0:
        #     for t in item['subgraph']:
        if len(subgraph) != 0:
            for t in subgraph:
                u, r, v = id2triple[t]
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]
                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab),
            "num_relation": rel_cnt * 2,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
        }

def make_gfm_first_input(item, id2triple): # item : {'Justin Bieber': [3512751, 3777290, 241649, 937262, 3476257]}
    kg_result = load_file(item, id2triple, inv_entity_vocab={}, inv_rel_vocab={})
    
    ent2id = kg_result["inv_entity_vocab"]
    rel2id = kg_result["inv_rel_vocab"]
    num_node = kg_result["num_node"]
    num_relations = kg_result["num_relation"]
    kg_triplets = kg_result["triplets"]
    
    train_target_edges = torch.tensor([[t[0], t[1]] for t in kg_triplets], dtype=torch.long).t()
    train_edges = train_target_edges
    train_target_etypes = torch.tensor([t[2] for t in kg_triplets])
    train_etypes = train_target_etypes

    kg_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node, target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations, rel2id=rel2id, ent2id=ent2id)
    kg_data = build_relation_graph(kg_data)
    
    return kg_data

def make_gfm_second_input(question, path, topic_entity, target_entity, text_encoder, kg_data, encode_path=False):
    question_entities_masks = []
    supporting_entities_masks = []
    ent2id = kg_data["ent2id"]
    num_nodes = kg_data["num_nodes"]

    not_filtered_target_entity = list(set(target_entity) & set(ent2id.keys()))
    target_entity = [item for item in not_filtered_target_entity if not item.startswith("m.") and not item.startswith("g.")]

    question_entities = [ent2id[x] for x in topic_entity if x in ent2id]
    supporting_entities = [ent2id[x] for x in target_entity if x in ent2id]
    question_entities_masks.append(entities_to_mask(question_entities, num_nodes))
    supporting_entities_masks.append(entities_to_mask(supporting_entities, num_nodes))

    question_entities_masks = torch.stack(question_entities_masks)
    supporting_entities_masks = torch.stack(supporting_entities_masks)
    if encode_path:
        question_embeddings = text_encoder(question + path).cpu()
    else:
        question_embeddings = text_encoder(question).cpu()

    question_dataset = datasets.Dataset.from_dict(
            {
                "question_embeddings": question_embeddings,
                "question_entities_masks": question_entities_masks,
                "supporting_entities_masks" : supporting_entities_masks,
            }
        ).with_format("torch")
    
    return question_dataset

def check_answer_whick_top(kg_data, target_entity):
    ent2id = kg_data["ent2id"]
    num_nodes = kg_data["num_nodes"]
    supporting_entities_masks = []

    target_entity = list(set(target_entity) & set(ent2id.keys()))

    supporting_entities = [ent2id[x] for x in target_entity]
    supporting_entities_masks.append(entities_to_mask(supporting_entities, num_nodes))
    supporting_entities_masks = torch.stack(supporting_entities_masks)

    return supporting_entities_masks

def batch_evaluate(pred, target, limit_nodes=None):
    num_target = target.sum(dim=-1)

    # answer2query = functional._size_to_index(num_answer)
    answer2query = torch.repeat_interleave(num_target)

    num_entity = pred.shape[-1]

    # in inductive (e) fb_ datasets, the number of nodes in the graph structure might exceed
    # the actual number of nodes in the graph, so we'll mask unused nodes
    if limit_nodes is not None:
        # print(f"Keeping only {len(limit_nodes)} nodes out of {num_entity}")
        keep_mask = torch.zeros(num_entity, dtype=torch.bool, device=limit_nodes.device)
        keep_mask[limit_nodes] = 1
        # keep_mask = F.one_hot(limit_nodes, num_entity)
        pred[:, ~keep_mask] = float("-inf")

    order = pred.argsort(dim=-1, descending=True)

    range = torch.arange(num_entity, device=pred.device)
    ranking = scatter_add(range.expand_as(order), order, dim=-1)

    target_ranking = ranking[target]
    # unfiltered rankings of all answers
    order_among_answer = variadic_sort(target_ranking, num_target)[1]
    order_among_answer = (
        order_among_answer + (num_target.cumsum(0) - num_target)[answer2query]
    )
    ranking_among_answer = scatter_add(
        variadic_arange(num_target), order_among_answer
    )

    # filtered rankings of all answers
    ranking = target_ranking - ranking_among_answer + 1
    ends = num_target.cumsum(0)
    starts = ends - num_target
    hard_mask = multi_slice_mask(starts, ends, ends[-1])
    # filtered rankings of hard answers
    ranking = ranking[hard_mask]

    return ranking, target_ranking