from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from utils import *
import numpy as np
from prompts import PROMPT_MULTI_ENT


class LLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, cache_dir='/home/huggingface').to(args.device)
        #self.model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, cache_dir='/home/huggingface', load_in_8bit = True, device_map="cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, cache_dir='/home/huggingface')
        self.logits = None  
        self.out_tokens = None
    def llm_call(self, input_text, max_new_token, printing=False, sub_Q=False,  get_logits=False):
        with torch.no_grad():
            input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'], max_new_tokens=max_new_token, pad_token_id=self.tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True)
            text = self.tokenizer.decode(outputs.sequences[0][input_ids['input_ids'].shape[1]:], skip_special_tokens=True)
            self.logits = outputs.scores
            self.out_tokens = np.array(outputs.sequences[0][input_ids['input_ids'].shape[1]:].tolist())
            
        if printing:
            print(text)
        if sub_Q:
            
            start = text.find('[')
            end = text.find(']')
            text_list =  smart_list_parser(text[start:end+1])
        
            if get_logits:
                tokens = outputs.sequences[0][input_ids['input_ids'].shape[1]:]
                top_scores, top_indices = torch.topk(self.logits[1][0], k=20)
                scores_tok20 = []
                for s, i in zip(top_scores, top_indices):
                    score = round(float(s.item()),3)
                    scores_tok20.append([score, self.tokenizer.decode(i)])

                if 7999 in tokens and 2074 in tokens:
                    ind = torch.where(tokens==7999)
                    start_idx = ind[0].tolist()[0]
                    start_idx +=1
                    ind = torch.where(tokens==2074)
                    end_idx = ind[0].tolist()[0]
                else:
                    start_idx, end_idx = 1, len(self.logits)-1
                return text_list, start_idx, end_idx, scores_tok20
            return text_list
        
        if 'Return :' in text:
            text = text.split('Return :')[1].strip()
        return text
    
    def get_first_big_div_Q(self, topic_box, total_original_q, dataset):
        is_printing = True
        if dataset == 'webqsp':
            en_qu_dict = dict()
            for item in topic_box:
                en_qu_dict[item] = total_original_q
 
            en_qu_dict, filtered_keys = get_en_qu_dict(en_qu_dict, total_original_q)
        
        else:
            input_text = PROMPT_MULTI_ENT.format(Q=total_original_q, topic=topic_box)
            text = self.llm_call(input_text, 250, printing=is_printing)
            en_qu_dict = extract_entity_question_chain(text)
            filtered_keys = list(en_qu_dict.keys())
        
        if list(set(topic_box) - set(filtered_keys)) == topic_box:
            filtered_keys = [topic_box[0]]
            del en_qu_dict
            en_qu_dict = dict()
            en_qu_dict[topic_box[0]] = [total_original_q]
 
        return en_qu_dict, filtered_keys
 

    def get_ans_temp(self, sub_Q):
        input_text = ANSWER_TEMPLATE.format(Q=sub_Q)
        out_form = self.llm_call(input_text, 10, printing=True)
        
        return out_form
    
    