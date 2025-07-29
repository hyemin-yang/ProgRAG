import os
import json
from tqdm import tqdm
import torch
import torch.utils.data

from typing import List
from collections import OrderedDict

from data_process import *
#from mpgfm_dataload import *
from config import args
from metric import hit_at_k, get_candidate_rank
from biencoder import CustomBertModel

from dataclasses import dataclass, asdict
from config import args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
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

class AttrDict:
    pass

@dataclass
class PredInfo:
    query: str
    pred_path: str
    pred_score: float
    topk_score_info: str
    rank: int
    correct: bool
    
class BertPredictor:

    def __init__(self, device):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False
        self.device = device
    def load(self, ckt_path, use_data_parallel=False):
        assert os.path.exists(ckt_path)
        ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        self.model = CustomBertModel(self.train_args)

        # DataParallel will introduce 'module.' prefix
        state_dict = ckt_dict['state_dict']
    
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        
        
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()

        if use_data_parallel and torch.cuda.device_count() > 1:
            print('Use data parallel predictor')
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.use_cuda = True
        elif torch.cuda.is_available():
            self.model.to(self.device)
            self.use_cuda = True
        print('Load model from {} successfully'.format(ckt_path))

        
    def load_default_model(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        print("=> creating model")
        self.model = CustomBertModel(self.args)
        self.model.eval()
        self.model.to(self.device)
        self.use_cuda = True
        
    def _setup_args(self):
        for k, v in args.__dict__.items():
            if k not in self.train_args.__dict__:
                print('Set default attribute: {}={}'.format(k, v))
                self.train_args.__dict__[k] = v
        print('Args used in training: {}'.format(json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)))
        
        args.is_test = True

    @torch.no_grad()
    def predict_by_examples(self):
        graph_path = "/home/hyemin/shared_data/webqsp/total_graph_webqsp.jsonl"
        test_path = "/home/hyemin/shared_data/webqsp/golden_path/test_goldenpath_newtotal_0416.jsonl"
        triple2id = "/home/hyemin/shared_data/webqsp/webqsp_triple2id.pkl"
        # graph_path = "/home/hyemin/shared_data/cwq/total_graph_cwq.jsonl"
        # test_path = "/home/hyemin/shared_data/cwq/golden_path/test_goldenpath_newtotal_0416.jsonl"
        # triple2id = "/home/hyemin/shared_data/cwq/cwq_triple2id.pickle"
        # #graph_path = "/home/hyemin/shared_data/cwq/test_max80000_subgraph.pkl"
        #rel2id = "/home/hyemin/shared_data/cwq/rel2id.pkl"
        
        batch_size=1
        max_num_neg = 500
        max_num_pos = 20
        
        dataset, triple2id, id2triple = new_load_data(test_path, graph_path, triple2id, max_num_neg, max_num_pos, tokenizer)
        test_dataset = Dataset(dataset)
        data_loader = torch.utils.data.DataLoader(
            test_dataset,
            num_workers=1,
            batch_size=batch_size,
            collate_fn=collate,
            shuffle=False)
        Hit1 = AverageMeter('Hit@1', ':6.2f')
        Hit3 = AverageMeter('Hit@3', ':6.2f')
        Hit10 = AverageMeter('Hit@10', ':6.2f')
        Hit20 = AverageMeter('Hit@20', ':6.2f')
        # Recall1 = AverageMeter('Hit@1', ':6.2f')
        # Recall3 = AverageMeter('Hit@3', ':6.2f')
        # Recall10 = AverageMeter('Hit@10', ':6.2f')
        
        false_samples = []
        for idx, batch_dict in enumerate(tqdm(data_loader)):
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict, self.device)
            
            scores, labels = self.model(**batch_dict)
            hits,_ = hit_at_k(scores, labels, topk=(1, 3, 10, 20))
            hit1, hit3, hit10, hit20 = hits[0], hits[1], hits[2], hits[3]
            

            
            # if hit10 == 0:
            #     line = OrderedDict()
            #     line['id'] = batch_dict['query_id']
            #     #line['query_hop'] = batch_dict['query_hop']
            #     sorted_ids = get_candidate_rank(scores, batch_dict['cand_triple_ids'])
            #     line['pred_top20_results'] = [id2triple[idx] for idx in sorted_ids[:20]]
            #     false_samples.append(line)
            # target_id = triple2id[('m.0bfmhy4', 'government.government_position_held.office_holder', 'Return J. Meigs, Jr.')]
            # rank, sorted_indices = get_candidate_rank(scores, batch_dict['cand_triple_ids'], target_id)
            # breakpoint()
            Hit1.update(hit1, batch_size)
            Hit3.update(hit3, batch_size)
            Hit10.update(hit10, batch_size)
            Hit20.update(hit20, batch_size)

        
        metric_dict = {'hit@1': round(Hit1.avg, 3),
                       'hit@3': round(Hit3.avg, 3),
                       'hit@10': round(Hit10.avg, 3),
                       'hit@20': round(Hit20.avg, 3),}
        
        print('Eval Results: {}'.format(json.dumps(metric_dict)))
        output_path = f"/home/hyemin/model/my_model/BACKPACK/ENT_PRUNER/cwq/hit10is0.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for line in false_samples:
                json.dump(line, f, ensure_ascii=False) 
                f.write("\n") 
            
                
device = 'cuda:1'
predictor = BertPredictor(device)
#ckt_path = '/home/hyemin/model/my_model/BACKPACK/ENT_PRUNER/webqsp/model_best.mdl'
#ckt_path = '/home/hyemin/shared_data/cwq/gfmpnet_inverse_entropy/model_best.mdl'
#ckt_path = '/home/hyemin/shared_data/cwq/gfmpnet_entropy/model_last.mdl'
ckt_path = '/home/hyemin/shared_data/roberta_model_best.mdl'
#ckt_path = '/home/hyemin/model/my_model/BACKPACK/ENT_PRUNER/cwq/alibaba_subgraph/checkpoint_epoch1.mdl'
#ckt_path = '/home/hyemin/model/my_model/BACKPACK/ENT_PRUNER/cwq/subgraph_add/checkpoint_epoch1.mdl'
#ckt_path = '/home/hyemin/model/my_model/BACKPACK/ENT_PRUNER/cwq/totalgraph_add/checkpoint_epoch1.mdl'
#ckt_path = '/home/hyemin/model/my_model/BACKPACK/ENT_PRUNER/webqsp/subgraph/model_last.mdl'
predictor.load(ckt_path=ckt_path)
#predictor.load_default_model(args)
predictor.predict_by_examples()