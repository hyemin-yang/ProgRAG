from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from utils import *
import numpy as np
from prompts import *

 
class LLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.is_GPT = args.is_GPT
        self.logits = 0
        self.outputs = 'None'
        self.input_ids = 'None'
        self.args = args

        if not self.is_GPT:
            self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, cache_dir='/home/huggingface')
            self.model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, cache_dir='/home/huggingface', device_map=self.device)
        else:
            self.model = 'GPT'

    def llm_call(self, input_text, max_new_token, task, printing=False):
        if not self.is_GPT:
            with torch.no_grad():
                self.input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.device)
                self.outputs = self.model.generate(input_ids=self.input_ids['input_ids'], attention_mask=self.input_ids['attention_mask'], max_new_tokens=max_new_token, pad_token_id=self.tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True)
                text = self.tokenizer.decode(self.outputs.sequences[0][self.input_ids['input_ids'].shape[1]:], skip_special_tokens=True)
                self.logits = self.outputs.scores

            if task in ['Total_Q', 'tog_relation']:
                pass
            elif task in ['subcheck', 'template', 'subcheck1']:
                text = text.split('\n')[0].strip()
            elif task == 'new_totalcheck':
                text = text.splitlines()
            else:
                try:
                    text = text.split('Return')[1].split('\n')[0].strip()[1:].strip()
                except:
                    text = "None"
                if task in ['relation']:
                    text = [item.strip() for item in text.split(',')]

        else:
            self.outputs, _ = ask_gpt4(input_text, self.args)
            if task in ['template', 'subcheck', 'subcheck1']:
                if self.outputs.startswith('```python'):
                    match = re.search(r'\[.*?\]', self.outputs)
                    self.outputs = match.group(0) if match else None
                text = self.outputs
            elif task in ['relation']:
                try:
                    text = self.outputs.split('Return')[1].split('\n')[0].strip()[1:].strip()
                    text = [item.strip() for item in text.split(',')]
                except:
                    text = self.outputs
            else:
                text = self.outputs
        
        if printing:
            print(text)

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

    
