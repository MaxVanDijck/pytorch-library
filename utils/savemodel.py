import torch

def saveModel(state, filename):
    print('Saving Checkpoint')
    torch.save(state, filename)