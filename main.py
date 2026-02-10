from llm import *
from prompt_matcher import *
from utils import *
from tqdm import tqdm
import gc
from GNN.gnn_utils import *
from GNN.gnn_text_encoder import *
from GNN.gnn_test import test
from GNN.nbfmodels import *
import pickle
import ast
from MPNet.predictor import LMPredictor
from itertools import chain
import argparse
import numpy as np
import os

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Our_model")
    parser.add_argument("--dataset", type=str, default='webqsp')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--gnn_device", type=str, default='cuda:0')
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--local_iter", type=int, default=2)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--return_entity_threshold", type=int, default=20)
    parser.add_argument("--return_entity_min_threshold", type=int, default=10)

    parser.add_argument("--webqsp_subgraph_path", type=str, default='./data/webqsp/webqsp_topic_graph.pkl')
    parser.add_argument("--cwq_subgraph_path", type=str, default='./data/cwq/cwq_topic_graph.pkl')
    parser.add_argument("--gnn_path", type = str, default = './ckpt/GNN')
    parser.add_argument("--lm_path", type=str, default = './ckpt/mpnet')
    parser.add_argument("--rel_ranker_path", type=str, default = './ckpt/sbert')

    # LLM related
    parser.add_argument("--is_GPT", action='store_true', default = False)
    parser.add_argument("--llm_model_path", type=str, default='google/gemma-2-9b-it')
    parser.add_argument("--gpt_model", type=str, default='gpt-4o-mini')
    parser.add_argument("--api_key", type=str, default='your api key')
    parser.add_argument("--do_uncertainty", action='store_true', default=False)
    parser.add_argument("--au_thres", type=float, default=1.55)
    parser.add_argument("--k_au", type=int, default=4) 
    parser.add_argument("--output_dir", type=str, default='output')
    args = parser.parse_args()
    
    ## 1) Load data
    dataset, total_graph, max_iter, id2triple, triple2id = get_dataset(args.dataset, args.split)
    tail_graph = get_tail_graph(total_graph)
    relation_graph = get_relation_graph(total_graph)

    ## 2) Load LLM
    model = LLM(args=args)
    
    ## 3) Load relation retriever & triple retriever (triple scorer, entity scorer) 
    GNN_device = args.gnn_device
    pruner = RelationRetriever(ckpt_path = args.rel_ranker_path, device = args.device, tail_graph = tail_graph, rel_graph = relation_graph, topk = args.topk)
    predictor = LMPredictor(GNN_device)
    text_encoder = GTELargeEN(GNN_device)
    
    if args.dataset == 'webqsp':
        with open(args.webqsp_subgraph_path, 'rb') as f:
            topic_graphs= pickle.load(f)
        gnn_model = GNNRetriever(entity_model=QueryNBFNet(input_dim=512, hidden_dims=[512, 512, 512]), rel_emb_dim=1024)
    else:
        with open(args.cwq_subgraph_path, 'rb') as f:
            topic_graphs= pickle.load(f)
        gnn_model = GNNRetriever(entity_model=QueryNBFNet(input_dim=512, hidden_dims=[512, 512, 512, 512, 512, 512]), rel_emb_dim=1024)
    
    predictor.load_model(ckt_path= f'{args.lm_path}/{args.dataset}.mdl')
    state = torch.load(f'{args.gnn_path}/{args.dataset}_GNN.pth', map_location="cpu")
    gnn_model.load_state_dict(state["model"])
    gnn_model.to(GNN_device)
   
    ## 4) initial setting
    out_file = f'{args.dataset}.jsonl'
    total_hit, total_f1 = [], [] 

    ## 5) start inference
    for inds in tqdm(range(len(dataset))): 
        total_original_q = dataset[inds]['question']
        topic_box = dataset[inds]['q_entity']
        path_map = defaultdict(list)
        all_subqs = []
        
        # Question decomposition
        en_qu_dict, filtered_keys = get_first_big_div_Q(model, topic_box, total_original_q, args.dataset)
        
        for topic_ent in filtered_keys:
          
            kg_data = "None"
            writer = toybox()
            graph_box = PathManager()
            cnt = 0      
            out_forms = ['object']

            if topic_ent not in relation_graph:
                gc.collect()
                torch.cuda.empty_cache()
                continue
                
            if topic_ent not in topic_graphs or topic_ent.startswith('g.'):
                graph = []
            else:
                item = topic_graphs[topic_ent]
                kg_data = make_gnn_first_input(item, id2triple)
                kg_data = kg_data.to(GNN_device)
                kg_data['rel_emb'] = get_emb(kg_data, args.dataset).to(GNN_device)
                graph = [id2triple[t] for t in item]
           
            
            original_q_box = en_qu_dict[topic_ent]

            if len(original_q_box) == 0:
                break
            original_q = original_q_box[0]

            #start sub-question answering
            while True:
                rel_ent_dict = dict()
                cnt += 1
                    
                if args.dataset == 'webqsp':
                    sub_Q = original_q
                elif args.dataset == 'cwq':
                    sub_Q = original_q

                    if len(en_qu_dict) != 1 and cnt == 1:
                        input_text = first_prompt_matching(cnt, original_q, topic_ent, writer)
                        sub_Qs = smart_list_parser(model.llm_call(input_text, 100, task='firstQ', printing=True))
                        original_q_box = sub_Qs
                    elif len(en_qu_dict) == 1 and len(original_q_box) == 1 and cnt == 1:
                        input_text = first_prompt_matching(cnt, original_q, topic_ent, writer)
                        sub_Qs = smart_list_parser(model.llm_call(input_text, 100, task='firstQ', printing=True))
                        original_q_box = sub_Qs
                        
                    try:
                        sub_Q = original_q_box[cnt-1]
                    except:
                        break
                    
                topic_ents = [topic_ent] if type(topic_ent) != list else topic_ent
                if type(topic_ent) != list:
                    topic_ent = [topic_ent]

                # Relation Retrieval
                cand_rel, cand_path = pruner.pruning(topic_ent, total_original_q, writer.ent_box)

                if args.dataset == 'webqsp':
                    out_forms = get_ans_temp(model, sub_Q)
                    try:
                        out_forms = ast.literal_eval(out_forms)
                    except:
                        out_forms = [out_forms]
                else:
                    out_forms = ['object']
                out_form = out_forms[0]

                # Relation Pruning
                if len(cand_rel) > 3:
                    input_text = rel_prompt_mathcing(args, topic_ent, sub_Q, writer, cand_rel, out_form)
                    retrieved_rel = model.llm_call(input_text, 500, task='relation', printing=True)
                elif len(cand_rel) > 0:
                    retrieved_rel = cand_rel
                else:
                    retrieved_rel = ["None"]
                writer.add_ent(topic_ent)
                all_subqs.append(sub_Q)
                    
                for iter in range(args.local_iter):
                    # Triple Retrieval
                    gnn_temp_rel_ent_dict, mpnet_temp_rel_ent_dict = dict(), dict()
                    gnn_id_temp_rel_ent_dict, mpnet_id_temp_rel_ent_dict= dict(), dict()
                    gnn_not_id_temp_rel_ent_dict, mpnet_not_id_temp_rel_ent_dict= dict(), dict()
                    mp_entity_to_triple = dict()
                    entity2prob = dict()
                    if retrieved_rel != ["None"]:
                        get_each_rel_end(cand_path, retrieved_rel, mpnet_temp_rel_ent_dict, gnn_temp_rel_ent_dict, graph, writer, mpnet=True, gnn=True)
                        total_triples = find_suited_trip(cand_path, list(mpnet_temp_rel_ent_dict.keys()))
                        graph_box.add_triples(total_triples)
                        
                        if len(gnn_temp_rel_ent_dict) != 0:
                            check_cnt = 0
                            for rel, cand_ent in gnn_temp_rel_ent_dict.items():
                                count = sum(1 for item in cand_ent if len(item) > 1 and item[1] == '.')
                                if count == len(cand_ent):
                                    check_cnt += 1
                                    gnn_id_temp_rel_ent_dict[rel] = gnn_temp_rel_ent_dict[rel]
                                else:
                                    gnn_not_id_temp_rel_ent_dict[rel] = gnn_temp_rel_ent_dict[rel]
                            # id
                            if check_cnt != 0:
                                id_topic_ent = list(set(value for sublist in gnn_id_temp_rel_ent_dict.values() for value in sublist))
                                id_topic_ents = [id_topic_ent] if type(id_topic_ent) != list else id_topic_ent
                                temp_id_cand_rel, cand_path = pruner.pruning(id_topic_ents, total_original_q, writer.ent_box)
                                gnn_new_temp_rel_ent_dict = dict()
                                retrieved_rel = temp_id_cand_rel
                                get_each_rel_end(cand_path, retrieved_rel,  None, gnn_new_temp_rel_ent_dict, graph, writer, mpnet=False, gnn=True)
                                
                                target_entity_list = list(set().union(*gnn_new_temp_rel_ent_dict.values()) | set().union(*gnn_not_id_temp_rel_ent_dict.values()))
                            else:
                                target_entity_list = list(set().union(*gnn_not_id_temp_rel_ent_dict.values()))
                            reasoning_path = topic_ents[0]
                            question_dataset = make_gnn_second_input(sub_Q, reasoning_path, topic_ents, target_entity_list, text_encoder, kg_data, encode_path=False)
                            outputs, entity2prob = test(gnn_model, kg_data, question_dataset, device=GNN_device)
                            
                            filtered_dict = {key: [val for val in values if val in outputs] for key, values in gnn_not_id_temp_rel_ent_dict.items() if any(val in outputs for val in values)}
                            if check_cnt != 0:
                                id_filtered_dict = {key: [val for val in values if val in outputs] for key, values in gnn_new_temp_rel_ent_dict.items() if any(val in outputs for val in values)}
                                filtered_dict.update(id_filtered_dict)

                        
                        if len(mpnet_temp_rel_ent_dict) != 0:
                            check_cnt = 0
                            for rel, cand_ent in mpnet_temp_rel_ent_dict.items():
                                count = sum(1 for item in cand_ent if len(item) > 1 and item[1] == '.')
                                if count == len(cand_ent):
                                    check_cnt += 1
                                    mpnet_id_temp_rel_ent_dict[rel] = mpnet_temp_rel_ent_dict[rel]
                                else:
                                    mpnet_not_id_temp_rel_ent_dict[rel] = mpnet_temp_rel_ent_dict[rel]
                            # id
                            if check_cnt != 0:
                                id_topic_ent = list(set(value for sublist in mpnet_id_temp_rel_ent_dict.values() for value in sublist))
                                id_topic_ents = [id_topic_ent] if type(id_topic_ent) != list else id_topic_ent
                                temp_id_cand_rel, cand_path = pruner.pruning(id_topic_ents, total_original_q, writer.ent_box)
                                mpnet_new_temp_rel_ent_dict = dict()
                                retrieved_rel = temp_id_cand_rel
                              
                                get_each_rel_end(cand_path, retrieved_rel, mpnet_new_temp_rel_ent_dict,  None, graph, writer, mpnet=True, gnn=False)
                                if len(mpnet_new_temp_rel_ent_dict) !=0:
                                    total_triples = find_suited_trip(cand_path, list(mpnet_new_temp_rel_ent_dict.keys()))
                                    graph_box.add_triples(total_triples)

                            mpnet_input_triples = graph_box.get_all_clean_chains(topic_ents)
                            
                            if len(mpnet_input_triples) != 0:

                                sorted_triples, sorted_scores = predictor.predict(sub_Q, mpnet_input_triples, path_map=None, k=len(mpnet_input_triples), chunk_size=1024)
                                mp_entity2prob = triple2prob(sorted_scores, sorted_triples)
                                mp_entity_to_triple = {trip[-1] : trip for trip in sorted_triples}

                                total_score_dict = dict()
                                for key, value in mp_entity2prob.items():
                                    if key in entity2prob:
                                        total_score_dict[key] = value  + entity2prob[key]    
                                    else:
                                        total_score_dict[key] = value 
                                #Repacking
                                sorted_dict = dict(sorted(total_score_dict.items(), key=lambda x: x[1], reverse=True))
                                sorted_cand_ent = list(sorted_dict.keys())
                                cand_ent_score = F.softmax(torch.tensor(list(sorted_dict.values())), dim=0).detach().cpu().tolist()

                                topp_list = []
                                topp_score_list = []
                                top_p = 0.9
                                temp_sum = 0
                                for i, item in enumerate(cand_ent_score):
                                    temp_sum += item
                                    topp_list.append(sorted_cand_ent[i])
                                    topp_score_list.append(item)
                                    if temp_sum > top_p:
                                        break
                                    
                                min_entity = args.return_entity_min_threshold
                                max_entity = args.return_entity_threshold 
                                if len(topp_list) < min_entity:
                                    top_k_cand_ent = sorted_cand_ent[:min_entity]
                                    top_k_cand_ent_score = cand_ent_score[:min_entity]
                                elif len(topp_list) > max_entity:
                                    top_k_cand_ent = topp_list[:max_entity]
                                    top_k_cand_ent_score = topp_score_list[:max_entity]
                                else:
                                    top_k_cand_ent = topp_list
                                    top_k_cand_ent_score = topp_score_list
                                
                               
                                top_k_cand_ent_score = F.softmax(torch.tensor(top_k_cand_ent_score), dim=0).detach().cpu().tolist()
   
                                triplets = []
                                for tail_entity, ent_score in zip(top_k_cand_ent, top_k_cand_ent_score):
                                    triplets.append(' '.join(mp_entity_to_triple[tail_entity])+'\t'+f'Candidate entity : ["{mp_entity_to_triple[tail_entity][-1]}"]')
                                    
                                input_text = SUBQUESTION_ANSWERING.format(Q=sub_Q + '?', T=triplets)
                                half_checking = model.llm_call(input_text, 600, task='subcheck', printing=True)
                                first_result  = smart_list_parser(half_checking)
                                half_result = first_result.copy() 
   
                                if args.do_uncertainty : 
                                    logit = model.logits[1][0].detach().cpu()
                                    temp_top_k_cand_ent = top_k_cand_ent + ["None"]
                                    ent_to_idx = {e: model.tokenizer.encode(e, add_special_tokens=False)[0] for e in temp_top_k_cand_ent}
                                    ent_logits = {e: model.logits[1][0, tid].item() for e, tid in ent_to_idx.items()}
                                    au, eu = cal_u(list(ent_logits.values()), min(5, len(ent_logits)))
                                    
                                    if au >= args.au_thres and 'None' not in half_result:
                                        second_half_result = top_k_cand_ent[:args.k_au]
                                        half_result += second_half_result
                                        half_result = set(half_result)
                                        if 'None' in half_result and len(half_result)>1:
                                            half_result -= set(['None'])
                                        half_result = list(half_result) 
                                        
                                    
                                rel_ent_dict = dict()
                                if half_result != ["None"]:
                                    rel_ent_dict = defaultdict(set)

                                    for key in half_result:
                                        entry = mp_entity_to_triple.get(key)
                                        if entry and len(entry) >= 3:
                                            relation, obj = entry[1], entry[2]
                                            rel_ent_dict[relation].add(obj)

                                    rel_ent_dict = {k: list(v) for k, v in rel_ent_dict.items()}

                                    cand_ent = half_result
                                    cand_ent1 = list(set().union(*rel_ent_dict.values()))
                                    option_map = {key : [value] for key, value in mp_entity_to_triple.items()}
                                    update_path_map(cand_ent1, option_map, path_map)                                    
     
                    if len(rel_ent_dict) != 0:
                        end_ent = list(set(chain.from_iterable(rel_ent_dict.values())))
                        checking = 'next'

                        if cnt == max_iter:
                            end_point = end_ent
                            set_box = list(rel_ent_dict.keys())
                            writer.add_all(all_subqs, set_box, topic_ent)
                            break
                    else:
                        if iter == args.local_iter:
                            break
                        rel_ent_dict = dict()
                        cand_rel = [x for x in cand_rel if x not in mpnet_temp_rel_ent_dict]
                        if len(cand_rel) > 3:
                            input_text = rel_prompt_mathcing(args, topic_ent, sub_Q, writer, cand_rel, out_form)
                            try:
                                retrieved_rel = model.llm_call(input_text, 500, task='relation', printing=True)
                            except:
                                retrieved_rel = ["None"]
                            if retrieved_rel != ["None"]:
                                checking = 'reset'

                        elif len(cand_rel) > 0:
                            retrieved_rel = cand_rel
                            checking = 'reset'
                        else:
                            retrieved_rel = ["None"]
                        
                        if type(topic_ent) != list:
                            _, cand_path = pruner.pruning([topic_ent], total_original_q, writer.ent_box)
    
                        else:
                            _, cand_path = pruner.pruning(topic_ent, total_original_q, writer.ent_box)

                    if checking == 'next':
                        set_box = list(rel_ent_dict.keys())
                        writer.add_all(all_subqs, set_box, topic_ent)
                        topic_ent = end_ent[0] if len(end_ent) == 1 else end_ent
                        break   
                
                if cnt == max_iter or (cnt >= 2 and len(rel_ent_dict) == 0):
                    break
                
        triplets = []
        totalq_input = []
        if len(path_map) !=0:
            final_entity2prob = dict()
            reasoning_path = topic_box[0]
            if filtered_keys[-1] in topic_graphs and not filtered_keys[-1].startswith('g.'):
                reasoning_path = topic_box[0]
                target_entity_list = list(path_map.keys())
                question_dataset = make_gnn_second_input(total_original_q, reasoning_path, topic_box, target_entity_list, text_encoder, kg_data, encode_path=True)
                final_outputs, final_entity2prob = test(gnn_model, kg_data, question_dataset, device=GNN_device)
                    
            mpnet_input_paths = []
            for tail, paths in path_map.items():
                for path in paths:
                    mpnet_input_paths.append(path)
                    
            sorted_triplets = []
            sorted_triples, sorted_scores = predictor.predict(total_original_q, mpnet_input_paths, path_map=None, k=len(mpnet_input_paths), chunk_size=1024)
            final_mp_entity2prob = triple2prob(sorted_scores, sorted_triples)
            ## gnn ## 
            final_total_score_dict = dict()
            for key, value in final_mp_entity2prob.items():
                if key in final_entity2prob:   
                    final_total_score_dict[key] = value + final_entity2prob[key] 
                else:
                    final_total_score_dict[key] = value
            
            final_sorted_dict = dict(sorted(final_total_score_dict.items(), key=lambda x: x[1], reverse=True))
            final_sorted_cand_ent = list(final_sorted_dict.keys())

            for tail in final_sorted_cand_ent:
                for path in path_map[tail]:
                    triplets.append(' '.join(path)+'\t'+f'Candidate entity : ["{tail}"]')
      
        
        #with repacking
        input_text = FINAL_ANSWER_PROMPT.format(Q=total_original_q, T=triplets)
        half_checking = model.llm_call(input_text, 600, task='subcheck', printing=True)
        end_point = smart_list_parser(half_checking)
        
        reasoning_paths = []
        for end in end_point:
            if end in path_map:
                reasoning_paths.extend(path_map[end])
        
        writer.add_rel(graph_box.get_all_relation_combinations_from_paths(reasoning_paths))
        writer.add_path(reasoning_paths)
        writer.add_qu(all_subqs)
        hit, f1 = write_log(os.path.join(args.output_dir, out_file), dataset, inds, total_original_q, end_point, writer.relation_list, writer.sub_questions)
        total_hit.append(hit)
        total_f1.append(f1)
        gc.collect()
        torch.cuda.empty_cache()   
        
    print("== Eval Results ==")
    print("EM:", sum(total_hit)/len(total_hit))
    print("F1:", sum(total_f1)/len(total_f1))
    print("right:", sum(total_hit))
    print("false:", len(total_hit)-sum(total_hit))
    print("#total:", len(total_hit))
