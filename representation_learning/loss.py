def simclr_loss(z1, z2, temperature=0.5, device='cpu'):
    bs, dim_feat = z1.shape
    reps = torch.cat([z1, z2], dim=0)
    reps = F.normalize(reps, dim=-1)

    logits = torch.matmul(reps, reps.T) / temperature

    # Filter out similarities of samples with themself
    mask = torch.eye(2 * bs, dtype=torch.bool, device=device)
    logits = logits[~mask]
    logits = logits.view(2 * bs, 2 * bs - 1)  # [2*b, 2*b-1]

    # The labels point from a sample in z1 to its equivalent in z2
    labels = torch.arange(bs, dtype=torch.long, device=device)
    labels = torch.cat([labels + bs - 1, labels])

    loss = F.cross_entropy(logits, labels)
    
    return loss