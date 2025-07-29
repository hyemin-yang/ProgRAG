from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments, losses
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import json
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from utils import get_relation_graph, get_tail_graph
from collections import defaultdict
import logging
import torch
from datasets import Dataset
import os
import random
from torch import Tensor, nn
from sentence_transformers.util import fullname



def split_path(paths,relation_graph, max_num_neg = 50):
    examples = []
    labels = []
    positives = defaultdict(set)
    for p in paths:
        for i in range(1,len(p),2):
            positives[p[i-1]].add(p[i])
            examples.append(p[i])
            labels.append(1)
            
    for key in positives.keys():
        negs = relation_graph[key] - positives[key]

        for neg in negs:
            examples.append(neg)
            labels.append(0)
 
    return examples , labels

def split_path_triple(paths,relation_graph, tail_graph, max_num_neg = 50):
    examples = []
    labels = []
    goldens = []
    positives = defaultdict(set)
    num_neg = 0 
    for pa in paths:
        goldens.append(pa[-1])
        for i in range(1,len(pa),2):
            positives[pa[i-1]].add(pa[i])
            examples.append(f'{pa[i-1]} {pa[i]} {pa[i+1]}')
            labels.append(1)
            goldens.append(pa[i-1])
    
    for key in positives.keys():
        rels = relation_graph[key] - positives[key]
        for rel in rels:
            negs = tail_graph[(key, rel)]
            for neg in negs:
                if num_neg>max_num_neg:
                    break
                if neg not in goldens:
                    examples.append(f'{key} {rel} {neg}')
                    labels.append(0)
                    num_neg +=1
        
    return examples , labels

def new_load_data(path, graph_path, path2=None, graph_path2=None):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print('Load {} examples from {}'.format(len(data), path))
    
    ddd = []
    with open(graph_path, 'r', encoding='utf-8') as f:
        for line in f:
            ddd.append(json.loads(line))
    triples = ddd[0]
    print('Load {} triples from {}'.format(len(triples), graph_path)) 
    rel_graph = get_relation_graph(triples)
    tail_graph = get_tail_graph(triples)
    train_size = int(len(data)*0.8)
    valid_size = len(data)-train_size
    train, valid = random_split(data, [train_size, valid_size])
        
    cnt = 0
    train_text1_list = []
    train_text2_list = []
    train_labels = []
    valid_text1_list = []
    valid_text2_list = []
    valid_labels = []
    
    for i in tqdm(range(len(train))):
        ex = data[i]
        if len(ex["golden_path"]) == 0:
            cnt += 1
            continue

        query = ex["question"]
        
        # examples, labels = split_path(ex["golden_path"], rel_graph)
        if len(ex["golden_path"]) == 0 :
            continue
        examples, labels = split_path_triple(ex["golden_path"], rel_graph, tail_graph)
        text1 = [query for _ in range(len(examples))]
        train_text1_list.extend(text1)
        train_text2_list.extend(examples)
        train_labels.extend(labels)
        
    print(f'Load {len(train_text1_list)} samples')
    
    
    

    for i in tqdm(range(len(valid))):
        ex = data[i]
        if len(ex["golden_path"]) == 0:
            cnt += 1
            continue

        query = ex["question"]
        #examples, labels = split_path(ex["golden_path"], rel_graph)
        examples, labels = split_path_triple(ex["golden_path"], rel_graph, tail_graph)
        text1 = [query for _ in range(len(examples))]
        valid_text1_list.extend(text1)
        valid_text2_list.extend(examples)
        valid_labels.extend(labels)

    print(f'Load {len(valid_text1_list)} samples')
    print(f'Skip {cnt} samples')
    print('Loading data has been finished')
    
    
    if path2:
        data = []
        with open(path2, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        print('Load {} examples from {}'.format(len(data), path2))
        
        ddd = []
        with open(graph_path2, 'r', encoding='utf-8') as f:
            for line in f:
                ddd.append(json.loads(line))
        triples = ddd[0]
        print('Load {} triples from {}'.format(len(triples), graph_path2)) 
        rel_graph = get_relation_graph(triples)

        train_size = int(len(data)*0.8)
        valid_size = len(data)-train_size
        train, valid = random_split(data, [train_size, valid_size])
        
        for i in tqdm(range(len(train))):
            ex = data[i]
            if len(ex["golden_path"]) == 0:
                cnt += 1
                continue

            query = ex["question"]
            #examples, labels = split_path(ex["golden_path"], rel_graph)
            examples, labels = split_path_triple(ex["golden_path"], rel_graph, tail_graph)
            text1 = [query for _ in range(len(examples))]
            train_text1_list.extend(text1)
            train_text2_list.extend(examples)
            train_labels.extend(labels)
            
        print(f'Load {len(train_text1_list)} samples')

        for i in tqdm(range(len(valid))):
            ex = data[i]
            if len(ex["golden_path"]) == 0:
                cnt += 1
                continue

            query = ex["question"]
            examples, labels = split_path(ex["golden_path"], rel_graph)
            text1 = [query for _ in range(len(examples))]
            valid_text1_list.extend(text1)
            valid_text2_list.extend(examples)
            valid_labels.extend(labels)

        print(f'Load {len(valid_text1_list)} samples')
    
    
    print(f'Skip {cnt} samples')
    print('Loading data has been finished')
    

    
    return {"text1": train_text1_list, "text2": train_text2_list, "label": train_labels}, \
        {"text1": valid_text1_list, "text2": valid_text2_list, "label": valid_labels}





os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = 'cuda'
DATA = 'webqsp'

model_save_path = "/REL_RETRIEVER"
gold_path = "/data/webqsp/train_goldenpath.jsonl"
graph_path = "/data/webqsp/total_graph_webqsp.jsonl"
gold_path2 = "/data/cwq/train_goldenpath.jsonl"
graph_path2 = "/data/cwq/total_graph_cwq.jsonl"

train_examples, valid_examples= new_load_data(gold_path, graph_path, gold_path2, graph_path2)
train_dataset = Dataset.from_dict(train_examples)
valid_dataset = Dataset.from_dict(valid_examples)

train_dataset = train_dataset.select_columns(["text1", "text2", "label"])
valid_dataset = valid_dataset.select_columns(["text1", "text2", "label"])

train_dataset.info.dataset_name = "MyTrainDataset"
valid_dataset.info.dataset_name = "MyValidDataset"

model  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device = device, num_labels=1) 

train_batch_size = 128
num_epochs = 5
pos_weight = torch.tensor([2.0])



# Loss function: BCEWithLogitsLoss
#train_loss = BinaryCrossEntropyLoss(model=model, pos_weight=pos_weight, hard_negative_weight = 1.5)
train_loss = losses.BinaryCrossEntropyLoss(model=model, pos_weight = pos_weight)
# Define training arguments
args = CrossEncoderTrainingArguments(
    output_dir=model_save_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    logging_steps=100,
    save_steps=5000,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=5000,
    logging_first_step=True,
    run_name="crossencoder_training",
    seed=42
)

# # Initialize the trainer
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,  # Use DataLoader for training
    eval_dataset=valid_dataset,   # Use DataLoader for validation
    loss=train_loss,
)

# trainer = CustomCrossEncoderTrainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,  # Use DataLoader for training
#     eval_dataset=valid_dataset,   # Use DataLoader for validation
#     loss=train_loss,
# )

# Train the model
trainer.train()

# Save the model
final_output_dir = f"{model_save_path}/final"
model.save_pretrained(final_output_dir)

# Optionally, push the model to Hugging Face Hub
# try:
#     model.push_to_hub("crossencoder-webqsp")
# except Exception as e:
#     logging.error(f"Error uploading model to the Hugging Face Hub: {e}")








