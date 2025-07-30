import argparse

parser = argparse.ArgumentParser(description='Training Bi-Encoder')
parser.add_argument('--pretrained-model', default= "microsoft/mpnet-base", type=str, metavar='N',
                    help='path to pretrained model')
parser.add_argument('--model-dir', default='ckpt', type=str, metavar='N',
                    help='path to model dir')
parser.add_argument('--max-to-keep', default=5, type=int, metavar='N',
                    help='max number of checkpoints to keep')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--pooling', default='mean', type=str, metavar='N',
                    help='bert pooling')
parser.add_argument('--temperature', default=0.07, type=float,  
                    help='temperature for supervised contrastive loss')
parser.add_argument('--lr', type=float, default=2e-5, #satkgc=2e-5
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
parser.add_argument('--train_path', default='./data/webqsp/train_goldenpath.jsonl')
parser.add_argument('--train_graph_path', default='./data/webqsp/total_graph_webqsp.json')
parser.add_argument('--train_path2', default='./data/cwq/train_goldenpath.jsonl')
parser.add_argument('--train_graph_path2', default='./data/cwq/total_graph_cwq.json')
parser.add_argument('--triple2id_1', default="./data/webqsp/webqsp_triple2id.json")
parser.add_argument('--triple2id_2', default="./data/cwq/cwq_triple2id.pickle")
parser.add_argument('--rel2id_path', default="./data/cwq/rel2id.pkl")
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--max_num_neg', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--max_num_pos', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--grad-clip', default=10.0, type=float, metavar='N',
                    help='gradient clipping')
parser.add_argument('--warmup', default=100, type=int, metavar='N',
                    help='warmup steps') #default=400
parser.add_argument('--lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
parser.add_argument('--output-dir', default='ckpt', type=str,
                    help='Lr scheduler to use')
args = parser.parse_args()
