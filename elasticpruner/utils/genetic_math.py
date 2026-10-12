# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F

def top_k_sim_logits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    k: int = int(32000 * 0.1),
    metric: str = "overlap",
    average: bool = True
):
    logits_q = logits_q.to(logits_p.device)
    if logits_p.shape != logits_q.shape:
        raise ValueError("logits_p and logits_q must have the same shape")

    V = logits_p.shape[-1]
    if k < 1 or k > V:
        raise ValueError(f"k must be in [1, {V}]")

    p = logits_p.reshape(-1, V)
    q = logits_q.reshape(-1, V)

    _, idx_p = torch.topk(p, k, dim=-1)  # (N, k), long
    _, idx_q = torch.topk(q, k, dim=-1)  # (N, k), long

    if metric == "overlap":
        eq = (idx_p.unsqueeze(2) == idx_q.unsqueeze(1))   # (N,k,k)
        inter = eq.any(dim=2).sum(dim=1).float()
        sims = inter.float() / float(k)

    elif metric == "jaccard":
        eq = (idx_p.unsqueeze(2) == idx_q.unsqueeze(1))   # (N,k,k)
        inter = eq.any(dim=2).sum(dim=1).float()          # (N,)
        union = 2 * k - inter                                # 2k - m
        sims = inter / union

    elif metric == "sim_rank":
        # Rank weighting: map tokens to ranks per row, then for the intersection min(1/(1+r_p), 1/(1+r_q))
        N = idx_p.size(0)
        ar = torch.arange(k, device=p.device).expand(N, k)
        # Gather is used because the token→rank dict must be built without a large table
        # To use the (N,k) tokens as keys, handle it with equality comparison instead of a hash table
        # Cost: O(N*k^2), but k is very small so it is fine
        rank_weight = 1.0 / (1.0 + ar.float())               # (N,k)
        # For each token (i), find its rank in q (brute-force matching)
        # eq: (N,k,k) -> get the position index in q and build the weights
        eq = (idx_p.unsqueeze(2) == idx_q.unsqueeze(1))      # (N,k,k) boolean
        # Rank weight in q for each item of p
        wq = torch.where(eq, rank_weight.unsqueeze(1).expand_as(eq).transpose(1,2),
                         torch.zeros_like(eq, dtype=rank_weight.dtype))
        # wq: for each of the k positions of p, the weight in q (0 if absent) → choose with max
        wq = wq.max(dim=2).values                             # (N,k)

        # Sum min(wp, wq) over the intersection items
        inter_mask = eq.any(dim=2)                            # (N,k)
        wp = rank_weight                                      # (N,k)
        contrib = torch.minimum(wp, wq) * inter_mask.float()  # (N,k)
        num = contrib.sum(dim=1)

        denom = torch.tensor(sum(1.0/(1+i) for i in range(k)), device=p.device)
        sims = (num / denom).clamp(0, 1)

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return sims.mean().item() if average else sims

@torch.no_grad()
def top_k_rwem(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    k: int = int(32000 * 0.01),
    average: bool = True,
    fast = True
):
    # logits_q = logits_q.to(logits_p.device)
    logits_p = logits_p.detach().to('cpu', non_blocking=False).contiguous()
    logits_q = logits_q.detach().to('cpu', non_blocking=False).contiguous()
    # logits_p = logits_p.detach().to('cuda').to(dtype=torch.float16)
    # logits_q = logits_q.detach().to('cuda').to(dtype=torch.float16)
    if logits_p.shape != logits_q.shape:
        raise ValueError("logits_p and logits_q must have the same shape")

    V = logits_p.shape[-1]
    if k < 1 or k > V:
        raise ValueError(f"k must be in [1, {V}]")

    p = logits_p.reshape(-1, V)
    q = logits_q.reshape(-1, V)

    _, idx_p = torch.topk(p, k, dim=-1)  # (N, k), long
    _, idx_q = torch.topk(q, k, dim=-1)  # (N, k), long
    if fast:
        score = rank_weighted_exact_match_fast(idx_p, idx_q)
    else:
        score = rank_weighted_exact_match(idx_p, idx_q)
    return score.cpu().mean().item()

@torch.no_grad()
def topk_cosine_sim_logits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    k: int = int(32000 * 0.1),
    average: bool = True,
    softmax_within_topk: bool = False,  # if True, normalize only the top-k values with softmax
):
    """
    1) Extract the top-k indices and values of p and q from each row
    2) Fill only those positions with the values in a zero-padded vector of length V (the rest is 0)
    3) Compute the cosine similarity of the two vectors

    Returns: scalar (float) if average=True, otherwise an (N,) tensor
    """
    # Device/shape check
    # logits_q = logits_q.to(logits_p.device)
    logits_p = logits_p.detach().to('cpu', non_blocking=False).contiguous()
    logits_q = logits_q.detach().to('cpu', non_blocking=False).contiguous()
    if logits_p.shape != logits_q.shape:
        raise ValueError("logits_p and logits_q must have the same shape")

    V = logits_p.shape[-1]
    if k < 1 or k > V:
        raise ValueError(f"k must be in [1, {V}]")

    # Flatten to (N, V)
    p = logits_p.reshape(-1, V)
    q = logits_q.reshape(-1, V)

    # top-k indices/values
    vals_p, idx_p = torch.topk(p, k, dim=-1)  # (N,k)
    vals_q, idx_q = torch.topk(q, k, dim=-1)  # (N,k)

    if softmax_within_topk:
        vals_p = F.softmax(vals_p, dim=-1)
        vals_q = F.softmax(vals_q, dim=-1)

    # Scatter into the zero-filled sparse vector
    vec_p = torch.zeros_like(p)
    vec_q = torch.zeros_like(q)
    vec_p.scatter_(1, idx_p, vals_p)
    vec_q.scatter_(1, idx_q, vals_q)

    # Cosine similarity (per row)
    # print(vec_p.shape, ", ", vec_q.shape)
    sims = F.cosine_similarity(vec_p.cpu(), vec_q.cpu(), dim=-1)  # (N,)

    return sims.cpu().mean().item() if average else sims


def rank_weighted_exact_match(idx_a: torch.Tensor,
                              idx_b: torch.Tensor,
                              scheme: str = "log",
                              alpha: float = 0.7,
                              dtype: torch.dtype = torch.float32):
    """
    idx_a, idx_b: (N, k) int64  — top-k indices for each row (in rank order)
    Returns: (N,) — score in [0,1]
    """
    assert idx_a.shape == idx_b.shape and idx_a.ndim == 2
    N, k = idx_a.shape

    # 1) Build the rank weights w_r (1-based rank)
    r = torch.arange(1, k+1, device=idx_a.device, dtype=dtype)
    if scheme == "log":
        w = 1.0 / torch.log2(r + 1.0)         # w_r = 1/log2(r+1)
    elif scheme == "harmonic":
        w = 1.0 / r                            # w_r = 1/r
    elif scheme == "geom":
        w = torch.pow(torch.tensor(alpha, device=idx_a.device, dtype=dtype), r-1)  # w_r = alpha^(r-1)
    else:
        raise ValueError("scheme must be one of {'log','harmonic','geom'}")

    # 2) (k,k) weight outer product W[i,j] = w_i * w_j
    W = torch.outer(w, w)  # (k,k)
    # print(W.shape)
    # 3) Rank match matrix M[n,i,j] = 1[a_i == b_j]
    M = (idx_a.unsqueeze(2) == idx_b.unsqueeze(1))  # (N,k,k)
    # print(M.shape)
    # 4) Weighted sum & normalization
    # Numerator: sum_{i,j} M[i,j] * W[i,j]
    num = (M * W).sum(dim=(1,2))                 # (N,)
    den = (w*w).sum()                             # scalar, maximum (identical/same order)
    score = (num / den).clamp_(0.0, 1.0)         # [0,1]
    return score

@torch.no_grad()
def rank_weighted_exact_match_fast(idx_a: torch.Tensor,
                                   idx_b: torch.Tensor,
                                   scheme: str = "log",
                                   alpha: float = 0.7,
                                   dtype: torch.dtype = torch.float32):
    """
    idx_a, idx_b: (N, k) int64 — each row: top-k indices sorted in rank order
    Returns: (N,) — score in [0,1]
    Memory-friendly: no large (N,k,k) or (k,k) tensors are created
    """
    assert idx_a.shape == idx_b.shape and idx_a.ndim == 2
    device = idx_a.device
    N, k = idx_a.shape

    # 1) Rank weights w_r (1-based rank)
    r = torch.arange(1, k+1, device=device, dtype=dtype)
    if scheme == "log":
        w = 1.0 / torch.log2(r + 1.0)
    elif scheme == "harmonic":
        w = 1.0 / r
    elif scheme == "geom":
        w = torch.pow(torch.tensor(alpha, device=device, dtype=dtype), r-1)
    else:
        raise ValueError("scheme must be one of {'log','harmonic','geom'}")

    # 2) Sort idx_b by value: the value array vals_b, and rank_b, the 'original rank' (1..k) of those values
    vals_b, order_b = torch.sort(idx_b, dim=1)             # (N,k), (N,k)
    rank_b = order_b.to(dtype=dtype) + 1                   # (N,k), original rank (1-based) of each value

    # 3) Find the candidate position pos in vals_b for each value of idx_a (binary search)
    pos = torch.searchsorted(vals_b, idx_a)                # (N,k), position 0..k
    pos_clamped = torch.clamp(pos, max=k-1)

    # 4) Check whether it is a real match (whether the values are identical)
    match = (vals_b.gather(1, pos_clamped) == idx_a)       # (N,k) bool

    # 5) Rank j in b (treated as 0 if there is no match)
    j = torch.where(match, rank_b.gather(1, pos_clamped), torch.zeros_like(pos_clamped, dtype=dtype))  # (N,k)
    j_idx = (j - 1).clamp_min(0).to(torch.long)            # 0 means no match

    # 6) Rank weight w_i of a, rank weight w_j of the matched rank in b  (w_j=0 for no match)
    #    a's ranks are already in order, so w_i = w (broadcast)
    w_i = w.unsqueeze(0).expand(N, -1)                     # (N,k)
    w_j = torch.where(match, w.index_select(0, j_idx.view(-1)).view(N, k), torch.zeros_like(w_i))  # (N,k)

    # 7) Numerator: sum_i w_i * w_j   /  Denominator: sum_r w_r^2
    num = (w_i * w_j).sum(dim=1)                           # (N,)
    den = (w * w).sum()                                    # scalar
    score = (num / den).clamp_(0.0, 1.0)                   # (N,)
    return score

if __name__=="__main__":
    tl = torch.load("../logits/logits_topk_exact_1756345844.9748611")
    score = top_k_rwem(tl[0], tl[1], fast=False)
    print(score)
    score = top_k_rwem(tl[0], tl[1], fast=True)
    print(score)
    
