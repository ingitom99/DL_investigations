import torch.optim
import torch.nn
from tqdm.auto import tqdm
import os
import numpy as np
import random
from model_makers import create_encoder, create_projection_head
from loss import simclr_loss

def train(loader, epochs, device):
    torch.manual_seed(0)
    random.seed(0)
    
    encoder = create_encoder()
    projection_head = create_projection_head()
    model = torch.nn.Sequential(encoder, projection_head)
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3, 
                                momentum=0.9, weight_decay=0.0001, 
                                nesterov=True)
    i = 0
    for k in range(epochs):
        losses = []
        for (x1, x2), _ in tqdm(loader):
            optimizer.zero_grad()

            x1 = x1.to(device)
            x2 = x2.to(device)

            z1 = model(x1)
            z2 = model(x2)
            loss = simclr_loss(z1, z2)
            loss.backward()
            optimizer.step()

            i += 1
            losses.append(loss.detach().item())

        print(f'epoch: {k}, loss: {np.mean(losses)}')

        # save model after each epoch
        os.makedirs('models', exist_ok=True)
        torch.save(encoder.state_dict(), f'models/model_{k}.pt')

    return encoder