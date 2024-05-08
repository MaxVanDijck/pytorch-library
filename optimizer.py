import torch
import math
from torch.optim.optimizer import Optimizer, required


class Ranger(Optimizer):

    def __init__(self, params, lr=2e-2):
        defaults = dict(lr=lr, step_counter=0, betas=(0.99, 0.99))
        super().__init__(params,defaults)
        self.radam_buffer = [[None,None,None] for ind in range(10)]

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)


    def step(self, closure=None):
        loss = None
        #Evaluate averages and grad, update param tensors
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:   # if first time to run...init dictionary with our desired entries
                    #if self.first_run_check==0:
                        #self.first_run_check=1
                        #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['1st_moment'] = torch.zeros_like(p_data_fp32)
                    state['2nd_moment'] = torch.zeros_like(p_data_fp32)

                    #look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['1st_moment'] = state['1st_moment'].type_as(p_data_fp32)
                    state['2nd_moment'] = state['2nd_moment'].type_as(p_data_fp32)

                #begin computations
                moment_1, moment_2 = state['1st_moment'], state['2nd_moment']
                beta1, beta2 = group['betas']

                #compute 1st_moment moving avg
                moment_1.mul_(beta1).add_(1 - beta1, grad)
                #compute 2nd_moment moving avg
                moment_2.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['step'] += 1

                step_size = 1
                denom = moment_2.sqrt().add_(group['eps']) + 1e-9
                p_data_fp32.addcdiv_(-step_size * group['lr'], moment_1, denom)
                
                # copy into optimizer
                p.data.copy_(p_data_fp32)

                #integrated look ahead...
                #we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer'] #get access to slow param tensor
                    slow_p.add_(self.alpha, p.data - slow_p)  #(fast weights - slow weights) * alpha
                    p.data.copy_(slow_p)  #copy interpolated weights to RAdam param tensor

        return loss
