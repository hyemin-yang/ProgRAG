import torch
from glob import glob
import os
from os.path import join
from tqdm import tqdm
from torch.utils import data as torch_data
from torch import nn
import numpy as np
from itertools import islice
from nbfmodels import *
from nbfutil import *
from nbflosses import *
from gnn_utils import *
import datasets
import json
from datasets import load_dataset
import gc

def test(model, test_datasets, device, return_metrics):
    print_topk = False
    world_size = get_world_size()
    rank = get_rank()
    batch_size = 1
    # process sequentially of test datasets
    all_metrics = {}
    all_mrr = []
    all_hit1 = []
    all_hit2 = []
    all_hit3 = []
    all_hit5 = []
    all_hit10 = []
    all_hit20 = []
    all_hit50 = []
    all_hit100 = []
    for data_name, q_data in test_datasets.items():
        test_data = q_data["data"]
        graph = q_data["graph"]
        sampler = torch_data.DistributedSampler(test_data, world_size, rank)
        test_loader = torch_data.DataLoader(test_data, batch_size, sampler=sampler)

        model.eval()
        ent_preds = []
        ent_targets = []

        entities_weight = None
        for batch in test_loader:
            ent2id =  q_data['graph']['ent2id']
            id2ent = {v : k for k, v in ent2id.items()}
            batch = {k : v.to(device) for k, v in batch.items()}
            graph = graph.to(device)
            graph['rel_emb'] = graph['rel_emb'].to(device)
            ent_pred, embedding = model(graph, batch, entities_weight=entities_weight)
            picks = len(ent_pred[0]) if len(ent_pred[0]) < 5 else len(ent_pred[0]) // 2
            if print_topk:
                try:
                    topkitems = torch.topk(ent_pred, k=picks)
                    indices = topkitems[1].tolist()[0]
                    result_list = [id2ent[idx] for idx in indices]
                except:
                    continue
            try:
                target_entities_mask = batch["supporting_entities_masks"]
                target_entities = target_entities_mask.bool()
                ent_ranking, target_ent_ranking = batch_evaluate(ent_pred, target_entities)
                # answer set cardinality prediction
                ent_prob = F.sigmoid(ent_pred)
                num_pred = (ent_prob * (ent_prob > 0.5)).sum(dim=-1)
                num_target = target_entities_mask.sum(dim=-1)
                ent_preds.append((ent_ranking, num_pred))
                ent_targets.append((target_ent_ranking, num_target))
                error='no'

            except:
                error = 'yes'

        if error == 'no':
            ent_pred, ent_target = gather_results(ent_preds[0], ent_targets[0], rank, world_size, device)

            ent_metrics = evaluate(ent_pred, ent_target, return_metrics)
            metrics = {}
            if rank == 0:
                for key, value in ent_metrics.items():
                    metrics[f"ent_{key}"] = value
                metrics["mrr"] = ent_metrics["mrr"]
        else:
            continue

        all_metrics[data_name] = metrics
        all_mrr.append(metrics["mrr"])
        all_hit1.append(metrics["ent_hits@1"])
        all_hit2.append(metrics["ent_hits@2"])
        all_hit3.append(metrics["ent_hits@3"])
        all_hit5.append(metrics["ent_hits@5"])
        all_hit10.append(metrics["ent_hits@10"])
        all_hit20.append(metrics["ent_hits@20"])
        all_hit50.append(metrics["ent_hits@50"])
        all_hit100.append(metrics["ent_hits@100"])
        torch.cuda.empty_cache()
    synchronize()

    all_avg_mrr = np.mean(all_mrr)
    print('mrr :', all_avg_mrr, end='\t')
    all_avg_hit1 = np.mean(all_hit1)
    print('hit1 :', all_avg_hit1, end='\t')
    all_avg_hit2 = np.mean(all_hit2)
    print('hit2 :', all_avg_hit2, end='\t')
    all_avg_hit3 = np.mean(all_hit3)
    print('hit3 :', all_avg_hit3, end='\t')
    all_avg_hit5 = np.mean(all_hit5)
    print('hit5 :', all_avg_hit5, end='\t')
    all_avg_hit10 = np.mean(all_hit10)
    print('hit10 :', all_avg_hit10, end='\t')
    all_avg_hit20 = np.mean(all_hit20)
    print('hit20 :', all_avg_hit20, end='\t')
    all_avg_hit50 = np.mean(all_hit50)
    print('hit50 :', all_avg_hit50, end='\t')
    all_avg_hit100 = np.mean(all_hit100)
    print('hit100 :', all_avg_hit100)

    return all_avg_mrr, all_avg_hit1, all_avg_hit2, all_avg_hit3, all_avg_hit5, all_avg_hit10, all_avg_hit20, all_avg_hit50, all_avg_hit100

is_test = False
load_model = False
device = get_device()
dataset_name = 'webqsp'
return_metrics = ['mrr', 'hits@1', 'hits@2', 'hits@3', 'hits@5', 'hits@10', 'hits@20', 'hits@50', 'hits@100']
text_encoder = GTELargeEN(device)

import pickle
with open(f'/data/{dataset_name}/{dataset_name}_triple2id.pkl', 'rb') as f:
    triple2id = pickle.load(f)
id2triple = {v : k for k, v in triple2id.items()}

if not is_test:
    with open(f'/data/{dataset_name}/small_subgraph.pkl', 'rb') as f:
        output= pickle.load(f)

    train_data = load_dataset(f"rmanluo/RoG-{dataset_name}", split='train')
    train_datasets = dict()

    for i, item in tqdm(enumerate(train_data), total=len(train_data)):
        good = 0
        if item['id'] in output:
            q_entitys = item['q_entity']
            for q_ent in q_entitys:
                if q_ent[0] not in ['g', 'm'] and q_ent[1] != '.':
                    good += 1
                    
            if good == len(q_entitys):
                # graph = output[item['id']]
                graph = item['graph']
                if len(graph) != 0:
                    target_entity_list = item['a_entity']
                    kg_data = make_gfm_first_input(graph, id2triple)
                    kg_data['rel_emb'] = get_rel_emb(kg_data, dataset_name)
                    reasoning_path = q_entitys[0]
                    question_dataset = make_gfm_second_input(item['question'], reasoning_path, q_entitys, target_entity_list, text_encoder, kg_data)
                    if sum(question_dataset['supporting_entities_masks'][0]).item() != 0:
                        train_datasets[i] = {"data": question_dataset, "graph": kg_data}
                
    valid_data = load_dataset(f"rmanluo/RoG-{dataset_name}", split='validation')
    valid_datasets = dict()
    for i, item in tqdm(enumerate(valid_data), total=len(valid_data)):
        for q_entity in item['q_entity']:
            if q_entity[0] not in ['g', 'm'] and q_entity[1] != '.':
                # graph = output[q_entity]
                graph = item['graph']
                target_entity_list = item['a_entity']
                kg_data = make_gfm_first_input(graph, id2triple)
                kg_data['rel_emb'] = get_rel_emb(kg_data, dataset_name)
                reasoning_path = q_entity
                question_dataset = make_gfm_second_input(item['question'], reasoning_path, [q_entity], target_entity_list, text_encoder, kg_data)
                if sum(question_dataset['supporting_entities_masks'][0]).item() != 0:
                    valid_datasets[i] = {"data": question_dataset, "graph": kg_data}

    rel_emb = train_datasets[0]['graph']['rel_emb'].shape[-1]
    if dataset_name == 'webqsp':
        model = GNNRetriever(entity_model=QueryNBFNet(input_dim=512, hidden_dims=[512, 512, 512]), rel_emb_dim=rel_emb)
    else:
        model = GNNRetriever(entity_model=QueryNBFNet(input_dim=512, hidden_dims=[512, 512, 512, 512, 512, 512]), rel_emb_dim=rel_emb)
    
    if load_model:
        pretrained_dict = torch.load(f"/data/{dataset_name}/model_best.pth", map_location="cpu")
        new_model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict["model"].items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        model.load_state_dict(new_model_dict)

    rank = get_rank()
    world_size = get_world_size()
    batch_size = 32

    train_dataloader_dict = {}
    for data_name, dataset in train_datasets.items():
        train_data = dataset["data"]
        sampler = torch_data.DistributedSampler(train_data, world_size, rank)
        train_loader = torch_data.DataLoader(train_data, batch_size, sampler=sampler)
        train_dataloader_dict[data_name] = train_loader
    data_name_list = list(train_dataloader_dict.keys())
    batch_per_epoch = len(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    loss_fn_list = []
    has_doc_loss = False
    for loss_cfg in ['ent_bce_loss', 'ent_pcr_loss']:
        if loss_cfg == "ent_bce_loss":
            loss_fn = BCELoss(adversarial_temperature=0.3)
            loss_fn_list.append(
            {
                "name": loss_cfg,
                "loss_fn": loss_fn,
                "weight": 0.4
            })
        elif loss_cfg == "ent_pcr_loss":
            loss_fn = ListCELoss()
            loss_fn_list.append(
            {
                "name": loss_cfg,
                "loss_fn": loss_fn,
                "weight": 0.6
            })
    
    parallel_model = model.to(device)

    batch_id = 0
    adversarial_temperature = 0.5
    best_result = float("-inf")

    for i in tqdm(range(0, 20)):
        epoch = i + 1
        parallel_model.train()

        losses = {loss_dict["name"]: [] for loss_dict in loss_fn_list}
        losses["loss"] = []

        for dataloader in train_dataloader_dict.values():
            dataloader.sampler.set_epoch(epoch)

        shuffled_data_name_list = np.random.permutation(data_name_list)
        
        for data_name in shuffled_data_name_list:
            train_loader = train_dataloader_dict[data_name]
            graph = train_datasets[data_name]["graph"]
            entities_weight = None

            for batch in train_loader:
                batch = {k : v.to(device) for k, v in batch.items()}
                graph = graph.to(device)
                graph['rel_emb'] = graph['rel_emb'].to(device)
                pred, embeds = parallel_model(graph, batch, entities_weight=entities_weight)
                target = batch["supporting_entities_masks"]

                loss = 0
                tmp_losses = {}
                for loss_dict in loss_fn_list:
                    loss_fn = loss_dict["loss_fn"]
                    weight = loss_dict["weight"]
                    single_loss = loss_fn(pred, target)
                    temp_loss = single_loss.detach().cpu().numpy()
                    tmp_losses[loss_dict["name"]] = single_loss.item()
                    loss += weight * single_loss
                tmp_losses["loss"] = loss.item()

                loss.backward() 
                optimizer.step()
                optimizer.zero_grad()

                for loss_log in tmp_losses:
                    losses[loss_log].append(tmp_losses[loss_log])

                batch_id += 1
                if batch_id % 10 == 0:
                    torch.cuda.empty_cache()

        state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
        torch.save(state, os.path.join(f"/data/{dataset_name}/mode_state_{i}.pth"))
        synchronize()
        torch.cuda.empty_cache()
        all_avg_mrr, all_avg_hit1, all_avg_hit2, all_avg_hit3,\
        all_avg_hit5, all_avg_hit10, all_avg_hit20, all_avg_hit50, all_avg_hit100 = test(model, valid_datasets, device=device, return_metrics=return_metrics)
        
        for loss_log in losses:
            print(sum(losses[loss_log]) / len(losses[loss_log]))

        if all_avg_hit3 > best_result:
            best_epoch = i
            best_result = all_avg_hit3
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, os.path.join(f"/data/{dataset_name}/model_best.pth"))
            print('updating')
        scheduler.step()
        torch.cuda.empty_cache()
 
