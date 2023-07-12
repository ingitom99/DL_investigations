import torch
from torch.utils.data import DataLoader

def compute_embeddings(dataset, encoder, device):
    dl = DataLoader(dataset, batch_size=32)
    X = []
    Y = []
    with torch.no_grad():
        for x, y, in dl:
            x = x.to(device)
            z = encoder(x)
            X.append(z)
            Y.append(y)
    X = torch.cat(X).cpu()
    Y = torch.cat(Y).cpu()
    return X, Y