import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

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
            # use_memory_efficient_attention=True
            ).to(device)
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

    def __call__(self, q_text):
        q_emb = self.embed([q_text])
        
        return q_emb
        
