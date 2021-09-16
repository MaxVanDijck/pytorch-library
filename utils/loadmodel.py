import torch

def loadModel(checkpoint, model, optimizer, lr):
    print('Loading Checkpoint')
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    #update learning-rate of old checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr