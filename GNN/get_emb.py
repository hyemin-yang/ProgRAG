import os
import torch
import json
import pickle
from tqdm import tqdm
import pydantic
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer



class EnvYaml(pydantic.BaseModel):
    num_threads: int
    seed: int

class TextEncoderYaml(pydantic.BaseModel):
    name: str

class EmbYaml(pydantic.BaseModel):
    env: EnvYaml
    entity_identifier_file: str
    text_encoder: TextEncoderYaml
    
def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    task = yaml_data.pop('task')
    assert task == 'emb'
    
    return EmbYaml(**yaml_data).model_dump()

class GTELargeEN:
    def __init__(self,
                 device,
                 normalize=True):
        self.device = device
        model_path = 'Alibaba-NLP/gte-large-en-v1.5'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            unpad_inputs=True,
            use_memory_efficient_attention=False).to(device)
        self.normalize = normalize

    @torch.no_grad()
    def embed(self, text_list):
        if len(text_list) == 0:
            return torch.zeros(0, 1024)
        
        batch_dict = self.tokenizer(
            text_list, max_length=8192, padding=True,
            truncation=True, return_tensors='pt').to(self.device)
        
        outputs = self.model(**batch_dict).last_hidden_state
        emb = outputs[:, 0]
        
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        
        return emb.cpu()

    def __call__(self, text_entity_list, relation_list):
        relation_embs = self.embed(relation_list)
        
        return relation_embs
    

def get_emb_question(subset, text_encoder, save_file):
    emb_dict = dict()
    for i in tqdm(range(len(subset))):
        id, q_text= subset[i]['id'], subset[i]['question']
        
        q_emb = text_encoder(q_text =q_text)

        emb_dict[id] = q_emb
    
    torch.save(emb_dict, save_file)


def get_emb_graph(triples, text_encoder, save_file_rel):
    ent_emb_dict = dict()
    rel_emb_dict = dict()
    entity_list = set()
    relation_list = set()
    
    for triple in tqdm(triples):
        entity_list.add(triple[0])
        relation_list.add(triple[1])
        entity_list.add(triple[2])
    
    entity_list = list(entity_list)
    relation_list = list(relation_list)

    text_entity_list = []
    non_text_entity_list = []
    for entity in entity_list:
        text_entity_list.append(entity)
    
    ent_path = os.path.join(os.path.dirname(save_file_rel), 'entity2id.pkl')
    rel_path = os.path.join(os.path.dirname(save_file_rel), 'rel2id.pkl')
    if not os.path.exists(ent_path):
        entity2id = dict()
        entity_id = 0
        for entity in text_entity_list:
            entity2id[entity] = entity_id
            entity_id += 1
        for entity in non_text_entity_list:
            entity2id[entity] = entity_id
            entity_id += 1
        with open(ent_path, 'wb') as f:
            pickle.dump(entity2id, f)
    else:
        with open(ent_path, 'rb') as fr:
            entity2id = pickle.load(fr)
   
    if not os.path.exists(rel_path):
        rel2id = dict()
        rel_id = 0
        for rel in relation_list:
            rel2id[rel] = rel_id
            rel_id += 1  
        with open(rel_path, 'wb') as f:
            pickle.dump(rel2id, f)               
    else:
        with open(rel_path, 'rb') as fr:
            rel2id = pickle.load(fr)
                 
    relation_embs = text_encoder(text_entity_list, relation_list)
    
    for ind, rel in enumerate(relation_list):
        rel_emb_dict[rel] = relation_embs[ind] 
    torch.save(rel_emb_dict, save_file_rel)
    
def main(args):
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/emb/gte-large-en-v1.5/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    torch.set_num_threads(config['env']['num_threads'])
    graph_file = args.graph_file
    graph = []
    with open(graph_file, 'r', encoding='utf-8') as f:
        for line in f:
            graph.append(json.loads(line))

    device = torch.device(args.device)
    
    text_encoder_name = config['text_encoder']['name']
    if text_encoder_name == 'gte-large-en-v1.5':
        text_encoder = GTELargeEN(device)
    else:
        raise NotImplementedError(text_encoder_name)
    
    emb_save_dir = f'dataset/{args.dataset}/emb'
    get_emb_graph(graph[0], text_encoder, os.path.join(emb_save_dir, 'relation.pth'))

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq', 'crlt', 'grailqa'], help='Dataset name')
    parser.add_argument('--graph_file', type=str, required=True, help='total graph file')
    parser.add_argument('--device', type=str, required=True, help='cuda device', default='cuda:0')
    
    args = parser.parse_args()
    main(args)
