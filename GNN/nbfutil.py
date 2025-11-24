import torch
from torch import distributed as dist
from torch_scatter import scatter_add, scatter_max, scatter_mean
import os
from .nbfvariadic import *

def get_rel_emb(temp, dataset): 
    relemb_path = 'data/{dataset}/relation.pth'
    rel_emb = torch.load(relemb_path)
    rel2id = temp['rel2id']
    rels = list(rel2id.keys())
    selected_tensors = [rel_emb[key] for key in rels if key in rel_emb]
    merged_tensor = torch.stack(selected_tensors, dim=0)
    del rel_emb

    return merged_tensor


def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(get_rank())
    else:
        device = torch.device("cpu")
    return device

def synchronize() -> None:
    if get_world_size() > 1:
        dist.barrier()

def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0

def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1

def get_entities_weight(ent2docs: torch.Tensor) -> torch.Tensor:
    frequency = torch.sparse.sum(ent2docs, dim=-1).to_dense()
    weights = 1 / frequency
    # Masked zero weights
    weights[frequency == 0] = 0
    return weights

def evaluate(pred, target, metrics):
    ranking, num_pred = pred
    answer_ranking, num_hard = target

    metric = {}
    for _metric in metrics:
        if _metric == "mrr":
            answer_score = 1 / ranking.float()
            query_score = variadic_mean(answer_score, num_hard)
        elif _metric.startswith("recall@"):
            threshold = int(_metric[7:])
            answer_score = (ranking <= threshold).float()
            query_score = (
                variadic_sum(answer_score, num_hard) / num_hard.float()
            )
        elif _metric.startswith("hits@"):
            threshold = int(_metric[5:])
            answer_score = (ranking <= threshold).float()
            # query_score = variadic_mean(answer_score, num_hard)
            query_score = answer_score.mean().unsqueeze(0)
        elif _metric == "mape":
            query_score = (num_pred - num_hard).abs() / (num_hard).float()
        else:
            raise ValueError(f"Unknown metric `{_metric}`")

        score = query_score.mean()
        name = _metric
        metric[name] = score.item()

    return metric


def variadic_area_under_roc(pred, target, size):
    """
    Area under receiver operating characteristic curve (ROC) for sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        pred (Tensor): prediction of shape :math:`(B,)`
        target (Tensor): target of shape :math:`(B,)`.
        size (Tensor): size of sets of shape :math:`(N,)`
    """

    index2graph = torch.repeat_interleave(size)
    _, order = variadic_sort(pred, size, descending=True)
    cum_size = (size.cumsum(0) - size)[index2graph]
    target = target[order + cum_size]
    total_hit = variadic_sum(target, size)
    total_hit = total_hit.cumsum(0) - total_hit
    hit = target.cumsum(0) - total_hit[index2graph]
    hit = torch.where(target == 0, hit, torch.zeros_like(hit))
    all = variadic_sum((target == 0).float(), size) * variadic_sum(
        (target == 1).float(), size
    )
    auroc = variadic_sum(hit, size) / (all + 1e-10)
    return auroc

def batch_evaluate(pred, target, limit_nodes=None):
    num_target = target.sum(dim=-1)

    # answer2query = functional._size_to_index(num_answer)
    answer2query = torch.repeat_interleave(num_target)

    num_entity = pred.shape[-1]

    # in inductive (e) fb_ datasets, the number of nodes in the graph structure might exceed
    # the actual number of nodes in the graph, so we'll mask unused nodes
    if limit_nodes is not None:
        # print(f"Keeping only {len(limit_nodes)} nodes out of {num_entity}")
        keep_mask = torch.zeros(num_entity, dtype=torch.bool, device=limit_nodes.device)
        keep_mask[limit_nodes] = 1
        # keep_mask = F.one_hot(limit_nodes, num_entity)
        pred[:, ~keep_mask] = float("-inf")

    order = pred.argsort(dim=-1, descending=True)

    range = torch.arange(num_entity, device=pred.device)
    ranking = scatter_add(range.expand_as(order), order, dim=-1)

    target_ranking = ranking[target]
    # unfiltered rankings of all answers
    order_among_answer = variadic_sort(target_ranking, num_target)[1]
    order_among_answer = (
        order_among_answer + (num_target.cumsum(0) - num_target)[answer2query]
    )
    ranking_among_answer = scatter_add(
        variadic_arange(num_target), order_among_answer
    )

    # filtered rankings of all answers
    ranking = target_ranking - ranking_among_answer + 1
    ends = num_target.cumsum(0)
    starts = ends - num_target
    hard_mask = multi_slice_mask(starts, ends, ends[-1])
    # filtered rankings of hard answers
    ranking = ranking[hard_mask]

    return ranking, target_ranking

def gather_results(pred, target, rank, world_size, device):
    # for multi-gpu setups: join results together
    # for single-gpu setups: doesn't do anything special
    ranking, num_pred = pred
    answer_ranking, num_target = target

    all_size_r = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_ar = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_p = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_r[rank] = len(ranking)
    all_size_ar[rank] = len(answer_ranking)
    all_size_p[rank] = len(num_pred)
    if world_size > 1:
        dist.all_reduce(all_size_r, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_ar, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_p, op=dist.ReduceOp.SUM)

    # obtaining all ranks
    cum_size_r = all_size_r.cumsum(0)
    cum_size_ar = all_size_ar.cumsum(0)
    cum_size_p = all_size_p.cumsum(0)

    all_ranking = torch.zeros(all_size_r.sum(), dtype=torch.long, device=device)
    all_num_pred = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)
    all_answer_ranking = torch.zeros(all_size_ar.sum(), dtype=torch.long, device=device)
    all_num_target = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)

    all_ranking[cum_size_r[rank] - all_size_r[rank] : cum_size_r[rank]] = ranking
    all_num_pred[cum_size_p[rank] - all_size_p[rank] : cum_size_p[rank]] = num_pred
    all_answer_ranking[cum_size_ar[rank] - all_size_ar[rank] : cum_size_ar[rank]] = (
        answer_ranking
    )
    all_num_target[cum_size_p[rank] - all_size_p[rank] : cum_size_p[rank]] = num_target

    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_pred, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_answer_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_target, op=dist.ReduceOp.SUM)

    return (all_ranking.cpu(), all_num_pred.cpu()), (
        all_answer_ranking.cpu(),
        all_num_target.cpu(),
    )
