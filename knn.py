import torch


def knn(x, k):
    
    inner = -2 * torch.matmul(x.transpose(2, 1), x) # batch multiplication between all pairs of points
    xx = torch.sum(x**2, dim=1, keepdim=True) # Squared Norm of each point
    pairwise_dist = -xx - inner - xx.transpose(2, 1) # Squared Distance formula bw two points
    idx = pairwise_dist.topk(k=k, dim=-1)[1] # For each point, list of k neighbors
    return idx 

def get_GraphFeatures(x, k=20, idx=None):
    
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k=k)
    
    idx_base = torch.arange(0, B, device=x.device).view(-1,1,1)*N
    idx = idx + idx_base
    idx = idx.view(-1)
    
    x = x.transpose(2, 1).contiguous()  # [B, N, C]
    feature = x.view(B * N, -1)[idx, :]  # [B * N * k, C]
    feature = feature.view(B, N, k, C)   # [B, N, k, C]
    
    x = x.view(B, N, 1, C).repeat(1, 1, k, 1)  # [B, N, k, C]
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature