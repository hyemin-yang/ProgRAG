from abc import ABC, abstractmethod
import torch
from typing import Any
from torch_scatter.composite import scatter_softmax
from torch.nn import functional as F
import torch.nn as nn

class BaseLoss(ABC):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        pass

class BCELoss(BaseLoss):
    def __init__(self, adversarial_temperature: float = 0, *args: Any, **kwargs: Any) -> None:

        self.adversarial_temperature = adversarial_temperature

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        is_positive = target > 0.5
        is_negative = target <= 0.5
        num_positive = is_positive.sum(dim=-1)
        num_negative = is_negative.sum(dim=-1)

        neg_weight = torch.zeros_like(pred)
        neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(
            num_positive
        )

        if self.adversarial_temperature > 0:
            with torch.no_grad():
                logit = pred[is_negative] / self.adversarial_temperature
                neg_weight[is_negative] = variadic_softmax(logit, num_negative)
        else:
            neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(
                num_negative
            )
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()
        return loss

class ListCELoss(BaseLoss):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        target_sum = target.sum(dim=-1)
        non_zero_target_mask = target_sum != 0  # Skip empty target
        target_sum = target_sum[non_zero_target_mask]
        pred = pred[non_zero_target_mask]
        target = target[non_zero_target_mask]
        pred_prob = torch.sigmoid(pred)  # B x N
        pred_prob_sum = pred_prob.sum(dim=-1, keepdim=True)  # B x 1
        loss = -torch.log((pred_prob / (pred_prob_sum + 1e-5)) + 1e-5) * target
        loss = loss.sum(dim=-1) / target_sum
        loss = loss.mean()
        return loss
    
def variadic_softmax(input, size):
    index2sample = torch.repeat_interleave(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    log_likelihood = scatter_softmax(input, index2sample, dim=0)
    return log_likelihood

class NCELoss(nn.Module):
    def __init__(self, eps=1e-12, temperature=0.07):
        super(NCELoss, self).__init__()
        self.eps = eps
        self.temperature = temperature

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        for i in range(scores.shape[0]):
            one_score = scores[i].unsqueeze(dim=0)
            label = labels[i].unsqueeze(dim=0)
            one_score = torch.sigmoid(one_score)
            score = one_score * 1/self.temperature
            exp_score = torch.exp(score)

            numerator = (exp_score * label).sum(dim=1)  # [B]
            denominator = exp_score.sum(dim=1)  # [B]
            loss = -torch.log((numerator / (denominator + self.eps)) + self.eps)
            if i == 0:
                losses = loss
            else:
                losses = torch.stack([losses, loss], dim=0)

        return losses.mean()
