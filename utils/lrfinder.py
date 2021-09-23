import numpy as np
import matplotlib.pyplot as plt

def LRFinder(model, loss_func, optimizer, dataloader, lr_range=(1e-6, 1), num_iter=50, repeat=1, device='cpu'):
    #Logging
    losses = []
    learning_rates = []
    
    #Generate learning rates
    lr_min, lr_max = lr_range
    lr_factor = np.exp(np.log(lr_max / lr_min) / num_iter)

    lrs = [lr_min]
    for i in range(num_iter):
        lrs.append(lrs[-1] * lr_factor)
        
    #Set model to device and save state
    model = model.to(device)
    
    batch = next(iter(dataloader))
    
    for i, lr in enumerate(lrs):
        #set lr
        for param in optimizer.param_groups:
            param['lr'] = lr
        
        #Get current loss
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        #zero the parameter gradients
        optimizer.zero_grad()

        #forward + backward + optimize
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        learning_rates.append(lr)
        
        #Record first loss value
        if i == 0:
            first_loss = loss.item()
        #If loss is starting to explode, break loop
        if loss.item() > first_loss * 2:
            break

    #Remove 10% of final values
    end_index = int(len(losses) * 0.9)
    losses = losses[0:end_index]
    learning_rates = learning_rates[0:end_index]
        
    #Find Local Minima
    min_value = min(losses)
    min_index = losses.index(min_value)

    #Set X-scale, learning rates to log
    plt.xscale("log")
    #Plot learning rates and losses
    plt.plot(learning_rates, losses)
    #Plot Minima
    plt.plot(learning_rates[min_index], losses[min_index], 'ro', ms=3) 
    plt.text(learning_rates[min_index], losses[min_index], 'Minima')
    #Find Recommended LR, minima lr / 10
    recommended_lr = learning_rates[min_index] / 10
    recommended_lr_yval = np.interp(learning_rates[min_index]/10, learning_rates, losses)
    #plot Recommended LR
    plt.plot(recommended_lr, recommended_lr_yval, 'ro', ms=3) 
    plt.text(recommended_lr, recommended_lr_yval, 'Recommended LR')

    #Print Info
    print('Min Loss: {:.4f}, LR at Min: {:.4e}, Recommended LR: {:.4e}'.format(losses[min_index], learning_rates[min_index], recommended_lr))