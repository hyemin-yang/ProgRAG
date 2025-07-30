from abc import ABC
from copy import deepcopy
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F

class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        
        self.query_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.target_bert = deepcopy(self.query_bert)
    
    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output
    

    def forward(self, 
            query_token_ids, query_mask, query_token_type_ids, 
            pos_token_ids, pos_mask, pos_token_type_ids,
            neg_token_ids, neg_mask, neg_token_type_ids,
            **kwargs) -> dict:

        B = query_token_ids.size(0)

        query_vector = self._encode(self.query_bert,
                                token_ids=query_token_ids,
                                mask=query_mask,
                                token_type_ids=query_token_type_ids)
        
        B, P, seq_len = pos_token_ids.shape
        pos_ids_flat = pos_token_ids.view(B * P, seq_len)
        pos_mask_flat = pos_mask.view(B * P, seq_len)
        pos_type_ids_flat = pos_token_type_ids.view(B * P, seq_len)
        
        pos_emb_flat = self._encode(self.target_bert,
                                    token_ids=pos_ids_flat,
                                    mask=pos_mask_flat,
                                    token_type_ids=pos_type_ids_flat)
        pos_emb = pos_emb_flat.view(B, P, -1)  # [B, P, D]
        

        _, N, seq_len = neg_token_ids.shape
        neg_ids_flat = neg_token_ids.view(B * N, seq_len)
        neg_mask_flat = neg_mask.view(B * N, seq_len)
        neg_type_ids_flat = neg_token_type_ids.view(B * N, seq_len)
        
        neg_emb_flat = self._encode(self.target_bert,
                                    token_ids=neg_ids_flat,
                                    mask=neg_mask_flat,
                                    token_type_ids=neg_type_ids_flat)
        neg_emb = neg_emb_flat.view(B, N, -1)  # [B, N, D]

        candidate_emb = torch.cat([pos_emb, neg_emb], dim=1)
        
        #scores = torch.bmm(query_vector.unsqueeze(1), candidate_emb.transpose(1, 2)).squeeze(1)
        scores = torch.matmul(query_vector.unsqueeze(1), candidate_emb.transpose(1, 2)).squeeze(1)
        scores*=1/self.args.temperature
        # scores = scores.masked_fill(candidate_mask == 0, -1e9)
        
        labels = torch.zeros_like(scores)
        labels[:, :P] = 1.0
        
        return scores, labels
    
    def encode_relation_embedding(self, relation_token_ids, relation_token_type_ids, relation_mask):
        candidate_embs = self._encode(self.target_bert,
                                    token_ids=relation_token_ids,
                                    mask=relation_mask,
                                    token_type_ids=relation_token_type_ids)
        return candidate_embs
    
    @torch.no_grad()
    def encode_triple_embedding(self, triple_token_ids,triple_token_type_ids, triple_mask):
        candidate_embs = self._encode(self.target_bert,
                                    token_ids=triple_token_ids,
                                    mask=triple_mask,
                                    token_type_ids=triple_token_type_ids)
        return {'triple_vectors': candidate_embs}
    @torch.no_grad()
    def encode_query(self, query_token_ids, query_mask, query_token_type_ids):
        query_embs = self._encode(self.query_bert,
                        token_ids=query_token_ids,
                        mask=query_mask,
                        token_type_ids=query_token_type_ids)
        return {'query_vector':query_embs}
     
def _pool_output(pooling: str, cls_output: torch.tensor, mask: torch.tensor,last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector



    
