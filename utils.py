import re
import torch.nn.functional as F
from sentence_transformers.cross_encoder import CrossEncoder
from collections import defaultdict, OrderedDict
from prompts import *
from datasets import load_dataset
import json
import networkx as nx
import ast
import torch
import pickle
import numpy as np
from scipy.special import digamma

def find_triples(graph, source, rel):
    temp = set()
    for t in graph:
        if source in t and rel in t:
            temp = set(t) | temp
    return temp

def find_suited_trip(cand_path, rels):
    rels_set = set(rels)
    temp_list = []

    for rel, path in cand_path.items():
        if rel in rels_set:
            temp_list.extend(path)
                
    return temp_list

class PathManager:
    def __init__(self):
        self.graph = defaultdict(list)  # src -> list of (relation, dst)

    def add_triples(self, triples):
        for src, rel, dst in triples:
            self.graph[src].append((rel, dst))

    def get_all_clean_chains(self, topic_ents):
        topic_set = set(topic_ents)
        results = set()

        def dfs(path, current_node, visited_entities):
            for rel, neighbor in self.graph[current_node]:
                if neighbor in visited_entities:
                    continue

                if isinstance(neighbor, str) and neighbor.startswith('m.'):
                    visited_entities.add(neighbor)
                    dfs(path, neighbor, visited_entities)
                    visited_entities.remove(neighbor)
                else:
                    if neighbor in topic_set:
                        continue

                    path.extend([rel, neighbor])
                    visited_entities.add(neighbor)
                    results.add(tuple(path[:]))
                    dfs(path, neighbor, visited_entities)

                    visited_entities.remove(neighbor)
                    path.pop()
                    path.pop()

        for ent in topic_ents:
            dfs([ent], ent, set([ent]))

        return [list(p) for p in results]

    def get_all_relation_combinations_from_paths(self, paths):
        all_relation_seqs = []

        for path in paths:
            relation_seq = []

            for i in range(0, len(path) - 2, 2):
                head = path[i]
                next_ent = path[i + 2]
                rel_used = path[i + 1]

                matched = False
                for rel1, mid in self.graph.get(head, []):
                    if isinstance(mid, str) and mid.startswith("m."):
                        for rel2, tail in self.graph.get(mid, []):
                            if tail == next_ent:
                                relation_seq.append(rel1)
                                relation_seq.append(rel2)
                                matched = True

                if not matched:
                    relation_seq.append(rel_used)

            all_relation_seqs.append(relation_seq)

        all_rel_list = [r for rel_seq in all_relation_seqs for r in rel_seq]
        all_rel_list = list(set(all_rel_list))
        
        return all_rel_list
        
def get_cand_rels(graph, topic_ent, last_topics):
    cal_rel = dict()
    last_rel = set()
    for t in graph:
        if len(set(topic_ent) & set(t)) > 0:
            if t[1] not in cal_rel:
                cal_rel[t[1]] = 0
            cal_rel[t[1]] += 1
            if len(set(last_topics) & set(t)) == 1:
                last_rel.add(t[1])
    last_rel = list(last_rel)

    cand_rel = set(cal_rel.keys())

    for rel in last_rel:
        if cal_rel[rel] == 1 or cal_rel[rel] == len(topic_ent):
            cand_rel.remove(rel)
    
    return list(cand_rel)

def get_ent_que(text):
    subq_values = re.findall(r'\[SUBQ\]\s(.*?)\s\[ENT\]', text)
    ent_matches = re.findall(r'\[ENT\]\s(.*?)\s*(?=\[ENT\]|\[SUBQ\]|\Z)', text)
    ent_keys = [match for match in ent_matches if not match.startswith('[ANS')]
    result = dict(zip(ent_keys, subq_values))

    return result

def get_one_ent_que(ent, text):
    temp_dict = dict()
    if text.count('[SUBQ]') == 2:
        ques = [text.split('[SUBQ]')[1].split('[ENT]')[0].strip(), text.split('[SUBQ]')[2].split('[ENT]')[0].strip()]
    elif text.count('[SUBQ]') == 1:
        ques = text.split('[SUBQ]')[1].split('[ENT]')[0].strip()
    else:
        ques = text
    temp_dict[ent] = ques
    
    return temp_dict

def clean_text(text):
    stripped_text = text.strip("'")
    return f"{stripped_text}"

class RelationRetriever:
    def __init__(self, ckpt_path, device = 'cuda:0', tail_graph = None, rel_graph=None, topk=50, elim_wh = True):
        self.model = CrossEncoder(ckpt_path, device=device)
        self.rel_graph = rel_graph
        self.device = device
        self.topk = topk
        self.tail_graph = tail_graph
        self.elim_wh = elim_wh
        self.question_words = ["what", "who", "where", "when", "which", "how"]

    def pruning(self, ent_list, question, visited_ent_list):
       
        if self.elim_wh :
            question = question.rstrip('?')    
            question = question.split()
            for i, que in enumerate(question):
                if que.lower() in self.question_words:
                    question.pop(i)
            question = " ".join(question)
       
        corpus = []
        corpus_rel = []
        corpus_path = defaultdict(list)
        for ent in ent_list:
            if ent not in self.rel_graph:
                continue
            for rel in self.rel_graph[ent]:
                tails = self.tail_graph[(ent, rel)]
               
                for tail in set(tails):
                    if tail in visited_ent_list:
                        continue
                    corpus.append(ent + ', ' + rel + ', ' + tail)
                    corpus_rel.append(rel)
                    corpus_path[rel].append([ent, rel, tail])
       
        if len(corpus) == 0:
            return [], None
       
        elif len(corpus_rel) < self.topk:
            return corpus_rel, corpus_path
       
        else:
            ranks = self.model.rank(question, corpus)
            set_rel = []
            topk_path = {}
            r = 0
            while len(set_rel) < self.topk and r < len(ranks):
                ind = ranks[r]['corpus_id']
                if corpus_rel[ind] not in set_rel:
                    set_rel.append(corpus_rel[ind])
                    topk_path[corpus_rel[ind]] = corpus_path[corpus_rel[ind]]
                r += 1
               
            topk_relation_list = set_rel
 
            return topk_relation_list, topk_path

def get_tail_graph(triples):
    graph = defaultdict(list)
    for t in triples:
        graph[(t[0], t[1])].append(t[2])
        graph[(t[2], t[1])].append(t[0])    
    return graph
               
def get_relation_graph(triples):
    rel_graph_set = {}
    for t in triples:
        if t[0] in rel_graph_set:
            rel_graph_set[t[0]].add(t[1])
        else:
            rel_graph_set[t[0]] = set()
            rel_graph_set[t[0]].add(t[1])
        if t[2] in rel_graph_set:
            rel_graph_set[t[2]].add(t[1])
        else:
            rel_graph_set[t[2]] = set()
            rel_graph_set[t[2]].add(t[1])
    
    return rel_graph_set

def get_dataset(dataset_name, split):
    dataset = load_dataset(f"rmanluo/RoG-{dataset_name}", split=split)
    graph = []
    file_path = f'/data/graphs/total_graph_{dataset_name}.jsonl'
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            graph.append(data)
    max_iter = 3 if dataset_name == 'cwq' else 1
    if dataset_name == 'webqsp':
        with open(f'/data/graphs/{dataset_name}_triple2id.pkl', 'rb') as f:
            triple2id = pickle.load(f)
    else:
        with open(f'/data/graphs/{dataset_name}_triple2id.pickle', 'rb') as f:
            triple2id = pickle.load(f)

    id2triple = {v : k for k, v in triple2id.items()}
    return dataset, graph[0],  max_iter, id2triple, triple2id

def get_en_qu_dict(en_qu_dict, total_original_q):
    keys = list(en_qu_dict.keys())
    filtered_keys = [item for item in keys if not item.startswith('[ANS')]
    filtered_keys = list(map(clean_text, filtered_keys))

    for old_key, new_key in zip(keys, filtered_keys):
        if old_key != new_key:  
            en_qu_dict[new_key] = en_qu_dict.pop(old_key)

    if len(filtered_keys) == 1:
        en_qu_dict[filtered_keys[0]] = [total_original_q]
    
    return en_qu_dict, filtered_keys

def write_log(outpath, dataset, inds, original_q, cand_ent_list, relation_list, sub_questions):
    line = OrderedDict()
    line['id'] = dataset[inds]['id']
    line['ind'] = inds
    line['question'] = original_q
    f1, pre, re = eval_f1(cand_ent_list, dataset[inds]['a_entity'])
    hit =  exact_match(dataset[inds]['a_entity'], cand_ent_list)
    line['hit'] = hit
    line['f1'] = f1
    line['precision'] = pre
    line['recall'] = re
    line['predict'] = cand_ent_list
    line['answer'] = dataset[inds]['a_entity']
    line['pred_relation'] = relation_list
    line['sub_questions'] = sub_questions
    print(line)
    with open(outpath, "a") as outfile:
        json_str = json.dumps(line)
        outfile.write(json_str + "\n")
        outfile.close()
    return hit, f1
class toybox:
    def __init__(self):
        self.ent_box = list()
        self.sub_questions = list()
        self.relation_list = list()
        self.path = list()
    
    def add_qu(self, sub_Qs):
        self.sub_questions = sub_Qs
    
    def add_rel(self, retrieved_rel):
        self.relation_list = retrieved_rel
    
    def add_ent(self, topic_ent):
        if type(topic_ent) == str:
            self.ent_box.append(topic_ent)
        else:
            self.ent_box += topic_ent
    
    def add_all(self, sub_Qs, retrieved_rel, topic_ent):
        self.add_qu(sub_Qs)
        self.add_rel(retrieved_rel)
        self.add_ent(topic_ent)

    def add_path(self, paths):
        self.path.append(paths)
        
    

def get_each_rel_end(cand_path, retrieved_rel, mpnet_rel_ent_dict, gnn_rel_ent_dict, graph, writer, mpnet=False, gnn=False):
    for relation in retrieved_rel:
        if relation in cand_path:
            paths = cand_path[relation]
            if mpnet:
                ents = [p[-1] for p in paths]
                ents = set(ents) - set(writer.ent_box)
                mpnet_rel_ent_dict[relation] = list(ents)
            if gnn:
                if len(graph)!=0:
                    ents = set()
                    for triple in paths:
                        if tuple(triple) in graph:
                            ents.add(triple[-1])
                    ents = ents - set(writer.ent_box)
                    gnn_rel_ent_dict[relation] = list(ents)
    

def checking_breakup(graph, q_entity, a_entity):
    q_cnt = 0
    a_cnt = 0
    for t in graph:
        if len(set(t) & set(a_entity)) != 0:
            q_cnt += 1
        if len(set(t) & set(q_entity)) != 0:
            a_cnt += 1
            
    if q_cnt == 0 or a_cnt == 0:
        return False
    
    return True

def path_to_string(path):
    result = ""
    for i, p in enumerate(path):
        if i == 0:
            h, r, t = p
            result += f"{h} -> {r} -> {t}"
        else:
            _, r, t = p
            result += f" -> {r} -> {t}"
            
    return result.strip()

def check_hit(answer, cand):
    if len(set(answer) & set(cand)) == 0:
        return 0
    return 1
    
def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    if len(answer) == 0:
        return 1, 1, 1
    matched = 0
    for a in answer:
        if a in prediction:
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def exact_match(response, answers):
    clean_result = ''.join(word.replace(' ', '') for word in response).lower()
    for ans in answers:
        clean_answer = ans.strip().replace(" ","").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return 1
    return 0 

def escape_inner_quotes(s):
    pattern = r"'([^']*?)'([^']*?)'"
    def replacer(match):
        full = match.group(0)
        inner1 = match.group(1).replace("'", "\\'")
        inner2 = match.group(2).replace("'", "\\'")
        return "'" + inner1 + inner2 + "'"
    
    s_fixed = re.sub(r"(?<=\w)'(?=\w)", r"\\'", s)
    return s_fixed


def make_unique_rel_ent_dict(not_id_temp_rel_ent_dict, outputs, rel_rank_dict):
    from collections import defaultdict

    value_to_keys = defaultdict(list)
    for key, values in not_id_temp_rel_ent_dict.items():
        for val in values:
            value_to_keys[val].append(key)

    from operator import itemgetter

    value_to_best_key = {}
    for val in outputs:
        if val in value_to_keys:
            keys = value_to_keys[val]
            best_key = min(keys, key=lambda k: rel_rank_dict.get(k, float('inf')))
            value_to_best_key.setdefault(best_key, []).append(val)

    return value_to_best_key



def smart_list_parser(s):
    s = s.strip()

    try:
        return json.loads(s)
    except:
        pass

    try:
        return ast.literal_eval(s)
    except:
        pass
    def fallback_parser(s):
        result = []
        i = 0
        n = len(s)
        while i < n:
            if s[i] == "'":
                i += 1
                buffer = ""
                while i < n:
                    if s[i] == "'" and (i + 1 == n or s[i + 1] in [',', ']']):
                        break
                    buffer += s[i]
                    i += 1
                result.append(buffer.strip())
            i += 1
        return result

    return fallback_parser(s)

def mpnet_topp(sorted_scores, sorted_triples, top_p, min_entity, max_entity):
    scores_np = F.softmax(sorted_scores, dim=0).detach().cpu().numpy()
   
    topp_list = []
    topp_scores = []
    temp_sum = 0
    for i, item in enumerate(scores_np):
        temp_sum += item.item()
        topp_list.append(sorted_triples[i])
        topp_scores.append(sorted_scores[i])
        if temp_sum > top_p:
            break
 
    if len(topp_list) < min_entity:
        topp_list = sorted_triples[:max_entity]
        topp_scores = sorted_scores[:max_entity]
    if len(topp_list)>max_entity:
        topp_list = topp_list[:max_entity]
        topp_scores = topp_scores[:max_entity]
    topp_scores = F.softmax(torch.tensor(topp_scores), dim=0).detach().cpu().numpy()
    return topp_list, topp_scores


def update_path_map(mapping_answer, option_map, path_map):
    for ent in mapping_answer:
        for t in option_map[ent]:
            if t[0] in path_map:
                # previous_paths = path_map.pop(t[0])
                previous_paths = path_map[t[0]]
                for prev_path in previous_paths:
                    combined_path = prev_path + t[1:]
                    path_map[t[-1]].append(combined_path)
            else:
                path_map[t[-1]].append(t)

def extract_entity_question_chain(text):
    parts = re.split(r"\bReturn:", text)
    if len(parts) < 2:
        return {} 
 
    text = parts[1]  
    lines = text.strip().split('\n')
 
    sub_qs = []
    entities = []
 
    for line in lines:
        if line.startswith("SUB-QUESTION"):
            m = re.match(r"SUB-QUESTION\d+:\s*(.+)", line)
            sub_qs.append(m.group(1).strip() if m else "")
        elif line.startswith("ENTITY"):
            m = re.match(r"ENTITY\d+:\s*(.*)", line)
            entities.append(m.group(1).strip() if m else "")
        # elif line.startswith("MAIN ENTITY"):
        #     m = re.match(r"MAIN ENTITY\d+:\s*(.*)", line)
        #     entities.append(m.group(1).strip() if m else "")
 
    result = defaultdict(list)
    for i, (entity, question) in enumerate(zip(entities, sub_qs)):
        entity = entity.strip().strip('"').strip("'") 
        question = question.strip()
 
        if not entity:
            if i > 0:
                prev_entity = entities[i - 1].strip().strip('"').strip("'") 
                result[prev_entity].append(question)
        elif re.fullmatch(r"\[ANS\d+\]", entity):
            if i > 0:
                prev_entity = entities[i - 1].strip().strip('"').strip("'") 
                result[prev_entity].append(question)
        else:
            result[entity].append(question)
 
    if len(entities) > 1 and entities[0].strip() == '':
        first_question = sub_qs[0] if len(sub_qs) > 0 else ''
        second_question = sub_qs[1] if len(sub_qs) > 1 else ''
        result[entities[1].strip()] = [first_question, second_question]
 
    return dict(result)



def get_first_big_div_Q(model, topic_box, total_original_q, dataset):
    is_printing = True
    if dataset == 'webqsp':
        en_qu_dict = dict()
        for item in topic_box:
            en_qu_dict[item] = [total_original_q]

        en_qu_dict, filtered_keys = get_en_qu_dict(en_qu_dict, total_original_q)
    
    else:
        input_text = PROMPT_MULTI_ENT.format(Q=total_original_q, topic=topic_box)
        text = model.llm_call(input_text, 250, task='Total_Q', printing=is_printing)
        en_qu_dict = extract_entity_question_chain(text)
        filtered_keys = list(en_qu_dict.keys())
    
    if list(set(topic_box) - set(filtered_keys)) == topic_box:
        filtered_keys = [topic_box[0]]
        del en_qu_dict
        en_qu_dict = dict()
        en_qu_dict[topic_box[0]] = [total_original_q]

    return en_qu_dict, filtered_keys


def get_ans_temp(model, sub_Q):
    input_text = REVISED_ANSWER_TEMPLATE.format(Q=sub_Q)
    
    out_form = model.llm_call(input_text, 10, task='template', printing=True)
    
    return out_form


def cal_u(logits, k):
    top_k = k
    if len(logits) < top_k:
        raise ValueError("Logits array length is less than top_k.")
    top_values = np.partition(logits, -top_k)[-top_k:]

    #calculate AU
    alpha = np.array([top_values])
    alpha_0 = alpha.sum(axis=1, keepdims=True)
    psi_alpha_k_plus_1 = digamma(alpha + 1)
    psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
    result = - (alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
    au = result.sum(axis=1)
    
    #calculate EU
    eu = top_k / (np.sum(np.maximum(0, top_values)) + top_k)
    #u = eu / au
    return au[0], eu

def top_k_mean(values, k):
        """Calculate the mean of the top k elements"""
        if values is None or len(values) == 0:
            return np.nan
        values = np.array(values)
        if len(values) <= k:
            return np.mean(values)
        top_k = np.partition(values, -k)[-k:]
        return np.mean(top_k)
    
    
import openai
from openai.error import Timeout, RateLimitError, APIConnectionError, APIError
import time

openai.api_key = "your api key"

def ask_gpt4(prompt):
    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                # model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
                request_timeout=1200,
                timeout=1200
            )
            return response.choices[0].message["content"]
        except Timeout as e:
            print(f"[OpenAI Timeout] Retry {attempt+1}/3: {e}")
            if attempt == 2:
                print("OpenAI request timed out 3 times. Returning empty string.")
                return ""
            else:
                time.sleep(10)
        except RateLimitError as e:
            print(f"[OpenAI RateLimitError] Retry {attempt+1}/3: {e}")
            if attempt == 2:
                print("Rate limit exceeded 3 times. Returning empty string.")
                return ""
            else:
                time.sleep(10)
        except APIConnectionError as e:
            print(f"[OpenAI APIConnectionError] Retry {attempt+1}/3: {e}")
            if attempt == 2:
                print("API connection failed 3 times. Returning empty string.")
                return ""
            else:
                time.sleep(10)
        except APIError as e:
            print(f"[OpenAI APIError] Retry {attempt+1}/3: {e}")
            if attempt == 2:
                print("API error occurred 3 times. Returning empty string.")
                return ""
            else:
                time.sleep(10)
        except Exception as e:
            print(f"[Other Error] Retry {attempt+1}/3: {e}")
            if attempt == 2:
                print("Unknown error occurred 3 times. Returning empty string.")
                return ""
            else:
                time.sleep(10)
 

