import torch
import torch.nn.functional as F
from gfm_utils import *

def test(model, kg_data, question_dataset, temp_retrieved_rel_endpoint_hit, answer_mask = None, is_topp = True, top_p = 0.9, hard_selection=True, threshod=0.5, do_entity_len_threshold=True, return_entity_threshold=10, device='cuda:0'):
    # max_top = 30
    max_top = 40
    eps = 1e-12
    avg_entropy = 0
    entity2prob = {}
    graph = [kg_data]
    batch = question_dataset[0]
    ent2id = kg_data['ent2id']
    id2ent = {v : k for k, v in ent2id.items()}
    '''[Data(edge_index=[2, 5], edge_type=[5], num_nodes=3, target_edge_index=[2, 5], target_edge_type=[5], num_relations=6, rel2id={location.country.languages_spoken=0,
    language.human_language.countries_spoken_in=1, location.country.official_language=2,}, relation_graph=Data(edge_index=[2, 18], edge_type=[18], num_nodes=6, num_relations=4), rel_emb=[3, 1024])]'''

    '''batch = {'question_embeddings': tensor([[ 0.0090, -0.0495,  0.0172,  ..., -0.0489,  0.0251, -0.0791]]), 'question_entities_masks': tensor([[1., 0., 0.]])}'''

    model.eval()

    entities_weight = None
    
    batch = {k : v.to(device) for k, v in batch.items()}
    ent_pred, _ = model(graph[0], batch, entities_weight=entities_weight)
    # ent_pred, emb = model(graph[0], batch, entities_weight=entities_weight)
    # emb = emb.detach().cpu().tolist()[0]
    # emb_dict = {value : emb[key] for key, value in id2ent.items()}
    # with open('temp_emb1.json', 'w') as f : json.dump(emb_dict, f)
    target_entities_mask = batch["supporting_entities_masks"].unsqueeze(0)
    target_num = int(sum(target_entities_mask[0]).item())
    # target_entities = target_entities_mask.bool()
    target_pred_score = torch.where(target_entities_mask == 0, torch.tensor(-float('inf'), device=device), ent_pred)
    
    if temp_retrieved_rel_endpoint_hit:
        answer_mask = answer_mask.to(device)
        answer_entities = answer_mask.bool()
        ent_ranking, target_ent_ranking = batch_evaluate(target_pred_score, answer_entities, limit_nodes=None)

    if is_topp:
        probs = F.softmax(target_pred_score, dim=1)
        node_entropy = -(probs * (probs + eps).log()).sum(dim=1)    
        avg_entropy = node_entropy.detach().item()
        scores_np = probs.detach().cpu().numpy()[0]
        candidate_list = []
        for i, score in enumerate(scores_np):
            candidate_list.append((score, 0 + i))

        sorted_candidates = sorted(candidate_list, key=lambda x: x[0], reverse=True)
        # sorted_scores = torch.tensor([score for score, _ in sorted_candidates])
        # sorted_indices = torch.tensor([idx for _, idx in sorted_candidates])
        entity2prob = {id2ent[ent] : prob for prob, ent in sorted_candidates}
        # 추가됨
        index_list = [idx for _, idx in sorted_candidates]
        #
    #     index_list = []
    #     temp_sum = 0
    #     for i, item in enumerate(sorted_scores):
    #         temp_sum += item.item()
    #         index_list.append(sorted_indices[i].item())
    #         if temp_sum > top_p:
    #             break

    #     if do_entity_len_threshold:
    #         if len(index_list) < return_entity_threshold:
    #             if target_num > return_entity_threshold:
    #                 topkitems = torch.topk(target_pred_score, k=return_entity_threshold)
    #             else:
    #                 topkitems = torch.topk(target_pred_score, k=target_num)
    #         else:
    #             if target_num > max_top:
    #                 topkitems = torch.topk(target_pred_score, k=max_top)
    #             else:
    #                 topkitems = torch.topk(target_pred_score, k=target_num)
    #         index_list = topkitems[1].tolist()[0]
    # else:
    #     if hard_selection:
    #         ent_prob = F.sigmoid(ent_pred) * target_entities_mask
    #         selected_index_tensor = ((ent_prob * (ent_prob > threshod)) > 0).int()
    #         index_list = (selected_index_tensor == 1).nonzero(as_tuple=True)[1].tolist()
        
    #     else:
    #         if target_num > return_entity_threshold:
    #             topkitems = torch.topk(target_pred_score, k = return_entity_threshold)
    #         else:
    #             topkitems = torch.topk(target_pred_score, k = target_num)
    #         index_list = topkitems[1].tolist()[0]
    
    selected_target = [id2ent[idx] for idx in index_list]

    if temp_retrieved_rel_endpoint_hit:
        return selected_target, target_ent_ranking, entity2prob, avg_entropy
    
    else:
        return selected_target, None, entity2prob, avg_entropy
            