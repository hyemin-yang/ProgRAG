from loss import SupConLoss, MPNCELoss
from biencoder import CustomBertModel
from transformers import AdamW, AutoTokenizer
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import random_split
from config import args
import logging
import random
from data_process import *
import time
from datetime import datetime
import shutil
import os
import glob
from metric import hit_at_k
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", show_time=True):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.show_time = show_time
        self.start_time = time.time()  # 시작 시각 저장

    def display(self, batch: int):
        entries = []
        # 현재 진행 배치 정보
        entries.append(self.prefix + self.batch_fmtstr.format(batch))
        # 각 meter 정보
        entries += [str(meter) for meter in self.meters]
        if self.show_time:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed = time.time() - self.start_time
            entries.append("Current Time: {}".format(now))
            entries.append("Elapsed: {:.2f}s".format(elapsed))
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
 
def save_checkpoint(state: dict, is_best: bool, filename: str):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.dirname(filename) + '/model_best.mdl')
    shutil.copyfile(filename, os.path.dirname(filename) + '/model_last.mdl')

def delete_old_ckt(path_pattern: str, keep=5):
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        print('Delete old checkpoint {}'.format(f))
        os.system('rm -f {}'.format(f))

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



           
class Trainer:
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        print("=> creating model")
        self.model = CustomBertModel(self.args)
        self._setup_training()
        self.criterion =  MPNCELoss().to(self.device)
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        #dataset=load_data(self.args.train_path, self.args.train_subgraph_path, self.args.triple2id, self.args.max_num_neg, self.tokenizer)
        #cwq, webqsp 한 번에 학습
        #dataset, self.triple2id, self.id2triple, self.rel2id, self.id2subgraph = new_load_data(self.args.train_path, self.args.train_graph_path, self.args.triple2id_1, '/home/hyemin/shared_data/cwq/rel2id.pkl', self.args.max_num_neg, self.args.max_num_pos, self.tokenizer)
        # self.encoded_relation = encoding_relations(list(self.rel2id.keys()), self.rel2id, self.model, self.tokenizer, self.device, batch_size=100)
        # torch.save(self.encoded_relation, '/home/hyemin/shared_data/cwq/mpnet_relation_embeddings.pt')
        #dataset,_,_  = allnew_load_data(self.args.train_path, self.args.train_graph_path, self.args.triple2id_1, self.args.rel2id_path, self.args.max_num_neg, self.args.max_num_pos, self.tokenizer)
        dataset,triple2id, id2triple  = new_load_data(self.args.train_path, self.args.train_graph_path, self.args.triple2id_1, self.args.max_num_neg, self.args.max_num_pos, self.tokenizer)
        #dataset = dataset + dataset2k
       
        train_size = int(len(dataset)*0.8)
        valid_size = len(dataset)-train_size
        train, valid = random_split(dataset, [train_size, valid_size])
        self.train_dataset = Dataset(train)
        self.valid_dataset = Dataset(valid)
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=args.workers, #추가하기.
                pin_memory=True,
                drop_last=True)

        self.valid_loader = torch.utils.data.DataLoader(
                    self.valid_dataset,                    
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate,
                    num_workers=args.workers,
                    pin_memory=True,
                    drop_last=True)
        
        num_training_steps = args.epochs * len(self.train_loader)
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None
    
    def _setup_training(self):
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model, device_ids = [0]).to("cuda:1")
        # elif torch.cuda.is_available():
        self.model.to(self.device)
        # else:
        #     print('No gpu will be used')
    def _create_lr_scheduler(self,num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                    num_warmup_steps=self.args.warmup,
                                                    num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                    num_warmup_steps=self.args.warmup,
                                                    num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)   
                
    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        metric_dict = self.eval_epoch(epoch)
        is_best = self.valid_loader and (self.best_metric is None or metric_dict['hit@1'] > self.best_metric['hit@1'])
        if is_best:
            self.best_metric = metric_dict

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.output_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.output_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.output_dir),
                       keep=self.args.max_to_keep)

    @torch.no_grad()
    def eval_epoch(self, epoch):
        if not self.valid_loader:
            return {}

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Hit@1', ':6.2f')
        top3 = AverageMeter('Hit@3', ':6.2f')
        top10 = AverageMeter('Hit@10', ':6.2f')
        top20 = AverageMeter('Hit@20', ':6.2f')
        batch_size = self.args.batch_size
        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict, self.device)
            
            # scores  = self.model(**batch_dict)
            # loss = self.criterion(scores, batch_dict['labels'])
            scores, labels  = self.model(**batch_dict)
            loss = self.criterion(scores, labels)
            losses.update(loss.item(), batch_size)
            
            #hits,_ = hit_at_k(scores,  batch_dict['labels'], topk=(1, 3, 10))
            hits,_ = hit_at_k(scores,  labels, topk=(1, 3, 10, 20))
            hit1, hit3, hit10, hit20 = hits[0], hits[1], hits[2], hits[3]
            
            top1.update(hit1, batch_size)
            top3.update(hit3, batch_size)
            top10.update(hit10, batch_size)
            top20.update(hit20, batch_size)

        metric_dict = {'hit@1': round(top1.avg, 3),
                       'hit@3': round(top3.avg, 3),
                       'hit@10': round(top10.avg, 3),
                       'hit@20': round(top20.avg, 3),
                       'loss': round(losses.avg, 3)}
        print('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        return metric_dict
    
    def train_epoch(self):
        
        batch_size = self.args.batch_size

        losses = AverageMeter('Loss', ':.4')
        
        for epoch in range(self.args.epochs):
            progress = ProgressMeter(len(self.train_loader), [losses], prefix="Epoch: [{}]".format(epoch))
            
            for i, batch_dict in enumerate(self.train_loader):
                
                #switch to train mode
                self.model.train()
                batch_dict = move_to_cuda(batch_dict, self.device)
                scores, labels  = self.model(**batch_dict)
                #scores = self.model(**batch_dict)
                loss = self.criterion(scores, labels)
                #loss = self.criterion(scores, batch_dict['labels'])
                
                losses.update(loss.item(), batch_size)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                
                if i % self.args.print_freq == 0:
                    progress.display(i)

            print('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

            self._run_eval(epoch=epoch)
           




def main():
    #ngpus_per_node = torch.cuda.device_count()
    device = 'cuda:2'
    ngpus_per_node = 1
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    #os.environ["CUDA_VISIBLE_DEVICES"]= "1" 
    print("Use {} gpus for training".format(ngpus_per_node))

    trainer = Trainer(args, device)
    print('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.train_epoch()


if __name__ == '__main__':
    main()