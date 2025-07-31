import torch
import json
from collections import OrderedDict
import heapq
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from .biencoder import CustomBertModel
from .data_process import Dataset

def move_to_cuda(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            #return maybe_tensor.cuda(non_blocking=True)
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def collate(batch_data, tokenizer):
    pad_token_id = tokenizer.pad_token_id
    input_ids_list = [torch.LongTensor(item['triple_token_ids']) for item in batch_data]
    token_type_ids_list = [torch.LongTensor(item['triple_token_type_ids']) for item in batch_data]

    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    padded_token_type_ids = pad_sequence(token_type_ids_list, batch_first=True, padding_value=0)
    attention_mask = (padded_input_ids != pad_token_id).long()

    return {
        'triple_token_ids': padded_input_ids,         # [B, seq_len]
        'triple_token_type_ids': padded_token_type_ids,  # [B, seq_len]
        'triple_mask': attention_mask                # [B, seq_len]
    }

          
def encoding_triples(triples, model, tokenizer, device, batch_size=1, ent_pruning=False, ROBERTA=False):
    examples = []
    for t in triples:
        encoded_triple = tokenizer(' '.join(t), add_special_tokens=True, max_length=200, 
                                        return_token_type_ids=True, truncation=True)
        examples.append({'triple_token_ids': encoded_triple['input_ids'],
            'triple_token_type_ids': encoded_triple['token_type_ids']})
    triple_dataset = Dataset(examples)   
    data_loader = torch.utils.data.DataLoader(
            triple_dataset,
            num_workers=0,
            batch_size = batch_size,
            collate_fn=lambda x: collate(x, tokenizer),
            shuffle=False)
    
    triple_tensor_list = []
    for idx, batch_dict in enumerate(data_loader):
        batch_dict = move_to_cuda(batch_dict, device)
        if ROBERTA:
            #for roberta - relation pruner
            output = model._encode(model.target_bert, batch_dict['triple_token_ids'], batch_dict['triple_mask'])
            triple_tensor_list.extend(output.detach().cpu().tolist())
            #breakpoint()
        else:
            output = model.encode_triple_embedding(**batch_dict)
            triple_tensor_list.append(output['triple_vectors'])

    if len(triple_tensor_list) !=0:
        if ROBERTA:
            all_triple_embs = torch.tensor(triple_tensor_list).to(device)
        else:
            all_triple_embs = torch.cat(triple_tensor_list, dim=0)
        return all_triple_embs
    else:
        return None


class AttrDict:
    pass

class LMPredictor:
    def __init__(self, device):
        self.model=None
        self.train_args=AttrDict()
        self.device = device
    
    def load_model(self, ckt_path):
        ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        #self.tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
        self.tokenizer = AutoTokenizer.from_pretrained(self.train_args.pretrained_model)
        self.model = CustomBertModel(self.train_args)
        print("=> creating model")
        
        # DataParallel will introduce 'module.' prefix
        state_dict = ckt_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()
        
        self.model.to(self.device)
        self.use_cuda = True
        print('Load model from {} successfully'.format(ckt_path))

        
    def _setup_args(self):
        # for k, v in args.__dict__.items():
        #     if k not in self.train_args.__dict__:
        #         print('Set default attribute: {}={}'.format(k, v))
        #         self.train_args.__dict__[k] = v
        print('Args used in training: {}'.format(json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)))
        
    def get_top_k_candidates(self, query_vector, all_triple_embs, k=10, chunk_size=1024):
        query_vector = query_vector.to(self.device)
        if query_vector.dim() == 1:
            query_vector = query_vector.unsqueeze(0)  # [1, D]
        N = all_triple_embs.size(0)
        
        if N <= chunk_size:
            all_triple_embs = all_triple_embs.to(self.device)
            scores = torch.matmul(query_vector, all_triple_embs.transpose(0, 1)).squeeze(0)  # [N]
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            return sorted_scores, sorted_indices
        
        candidate_list = []  
        start = 0
        while start < N:
            end = min(start + chunk_size, N)
            chunk_embs = all_triple_embs[start:end].to(self.device)
            # query_vector: [1, D], chunk_embs: [chunk_size, D]
            scores = torch.matmul(query_vector, chunk_embs.transpose(0, 1)).squeeze(0)  # [chunk_size]
            scores_np = scores.detach().cpu().numpy()
            for i, score in enumerate(scores_np):
                candidate_list.append((score, start + i))
            start = end

        sorted_candidates = sorted(candidate_list, key=lambda x: x[0], reverse=True)
        sorted_scores = torch.tensor([score for score, idx in sorted_candidates])
        sorted_indices = torch.tensor([idx for score, idx in sorted_candidates])
        return sorted_scores, sorted_indices

    @torch.no_grad()
    def predict(self, query, triples, path_map=None, k=10, chunk_size=128, ent_pruning=True, rel_pruning=False):
        if path_map:
            input_triples = []
            for t in triples:
                if t[0] in path_map:    
                    previous_paths = path_map[t[0]]
                    for prev_path in previous_paths:
                        combined_path = prev_path + t[1:]
                        input_triples.append(combined_path)
                else:
                    input_triples.append(t)
            
        else:
            input_triples = triples  
        encoded_query = self.tokenizer(query, add_special_tokens=True, max_length=200,
                                        return_token_type_ids=True, truncation=True)
        
        pad_token_id = self.tokenizer.pad_token_id
        query_ids = torch.LongTensor(encoded_query['input_ids']).unsqueeze(0).to(self.device)
        query_mask = (query_ids != pad_token_id).long().to(self.device)
        query_token_type_ids = torch.LongTensor(encoded_query['token_type_ids']).unsqueeze(0).to(self.device)
        query_emb = self.model.encode_query(query_token_ids=query_ids,query_mask=query_mask,
                                            query_token_type_ids=query_token_type_ids)['query_vector']
        all_triple_embs = encoding_triples(input_triples, self.model, self.tokenizer, self.device, batch_size=chunk_size)
        if all_triple_embs !=None:            
            sorted_scores, sorted_indices = self.get_top_k_candidates(query_emb, all_triple_embs, chunk_size=chunk_size)
            #sorted_triples = [triples[i] for i in sorted_indices.tolist()]
            
            sorted_triples = [input_triples[i] for i in sorted_indices]
            if ent_pruning:
                distinct_triples = []
                distinct_sorted_scores = []
                seen_tails = set()
                for idx, triple in enumerate(sorted_triples):
                    tail = triple[-1]  
                    if tail not in seen_tails:
                        #distinct_triple_embs.append(all_triple_embs[sorted_indices[idx]])
                        distinct_triples.append(triple)
                        distinct_sorted_scores.append(sorted_scores[idx])
                        seen_tails.add(tail)
                    if len(distinct_triples) == k:
                        break
                return distinct_triples, torch.tensor(distinct_sorted_scores) #, distinct_triple_embs
            
            if rel_pruning:
                distinct_triples = []
                distinct_sorted_scores = []
                seen_relations = set()
                for idx, triple in enumerate(sorted_triples):
                    relation = triple[1]  
                    if relation not in seen_relations:
                        distinct_triples.append(triple)
                        distinct_sorted_scores.append(sorted_scores[idx])
                        seen_relations.add(relation)
                    if len(distinct_triples) == k:
                        break
                    
                return distinct_triples, torch.tensor(distinct_sorted_scores)
        else:
            return [], []

