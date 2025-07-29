import torch
import torch.nn.functional as F

def retrieve_top_k(query_embeddings, candidate_embeddings, k=5):
    cos_sim = F.cosine_similarity(query_embeddings.unsqueeze(1), candidate_embeddings.unsqueeze(0), dim=-1)
    topk_scores, topk_indices = torch.topk(cos_sim, k=k, dim=1)
    return topk_indices, topk_scores

def recall_at_k(scores: torch.Tensor, labels: torch.Tensor, topk) -> float:
    """
    scores: [B, P+N]
    labels: [B, P+N] (1이면 positive, 0이면 negative)
    """
    with torch.no_grad():
        B = scores.size(0)
        maxk = max(topk)
        # 상위 maxk개 후보 인덱스 구하기: shape [B, maxk]
        _, topk_indices = scores.topk(maxk, dim=1)

        # 각 k에 대해 correct 횟수를 저장할 리스트
        correct_counts = [0 for _ in topk]

        # 배치 내 각 샘플별로 처리
        for i in range(B):
            # topk에 포함된 각 k값에 대해
            for idx, k in enumerate(topk):
                # 이 샘플에서 상위 k개의 후보 인덱스
                selected_idx = topk_indices[i, :k]
                # selected_idx 중 positive(1)가 하나라도 있으면 recall 성공
                if (labels[i, selected_idx] == 1).any():
                    correct_counts[idx] += 1

        # 각 k에 대한 recall 계산
        recalls = []
        for idx, k in enumerate(topk):
            recall_k = 100.0 * correct_counts[idx] / B
            recalls.append(recall_k)

        # topk에 여러 값이 있으면 리스트, 단 하나라면 스칼라로 반환
        if len(topk) == 1:
            return recalls[0]
        else:
            return recalls


def hit_at_k(scores: torch.Tensor, labels: torch.Tensor, topk) -> float:
    """
    scores: [B, P+N]
    labels: [B, P+N] (1이면 positive, 0이면 negative)
    """
    with torch.no_grad():
        # labels = labels.unsqueeze(0)
        # scores = scores.unsqueeze(0)
        B, num_items = scores.shape
        maxk = max(topk)

        # topk 값이 num_items보다 큰 경우를 처리
        valid_topk = [k for k in topk if k <= num_items]
        invalid_topk = [k for k in topk if k > num_items]

        if valid_topk:
            _, topk_indices = scores.topk(max(valid_topk), dim=1)

            correct_counts = {k: 0 for k in valid_topk}
            for i in range(B):
                for k in valid_topk:
                    selected_idx = topk_indices[i, :k]
                    if (labels[i, selected_idx] == 1).any():
                        correct_counts[k] += 1

            # 각 k에 대한 hit 계산
            hits = {k: (correct_counts[k] / B) * 100 for k in valid_topk}
        else:
            hits = {}

        # 불가능한 topk는 100% hit으로 처리
        for k in invalid_topk:
            hits[k] = 100.0

        # 반환 형태 조정
        if len(topk) == 1:
            return hits[topk[0]], topk_indices if valid_topk else None
        else:
            return [hits[k] for k in topk], topk_indices if valid_topk else None
        
def get_candidate_rank(scores: torch.Tensor, cand_triple_ids: list, target_id=None) -> int:
    # 만약 scores가 2D 텐서라면 (batch 크기가 1이라고 가정)
    if scores.dim() == 2:
        scores = scores[0]  # [P+N] shape의 1D 텐서로 변환

    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    sorted_ids = [cand_triple_ids[i] for i in sorted_indices.tolist()] 
    if target_id:
        try:
            rank = sorted_ids.index(target_id) + 1
        except ValueError:
            rank = -1
        return rank, sorted_indices
    else:
        return sorted_ids
def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    