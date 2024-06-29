import torch
import math
from torch.optim.optimizer import Optimizer, required


class MyOptim(Optimizer):

    def __init__(self, params, lr=2e-2):
        defaults = dict(lr=lr, step_counter=0, betas=(0.99, 0.9))
        super().__init__(params,defaults)
        self.radam_buffer = [[None,None,None] for ind in range(10)]
        self.max_loss = float('-inf')
        self.ma_loss = 0

    def __setstate__(self, state):
        print("set state called")
        super(MyOptim, self).__setstate__(state)


    def unit_norm(self, x):
        """ axis-based Euclidean norm"""
        # verify shape
        keepdim = True
        dim = None

        xlen = len(x.shape)
        # print(f"xlen = {xlen}")

        if xlen <= 1:
            keepdim = False
        elif xlen in (2, 3):  # linear layers
            dim = 1
        elif xlen == 4:  # conv kernels
            dim = (1, 2, 3)
        else:
            dim = tuple(
                [x for x in range(1, xlen)]
            )  # create 1,..., xlen-1 tuple, while avoiding last dim ...

        return x.norm(dim=dim, keepdim=keepdim, p=2.0)

    def agc(self, p):
        """clip gradient values in excess of the unitwise norm.
        the hardcoded 1e-6 is simple stop from div by zero and no relation to standard optimizer eps
        """

        # params = [p for p in parameters if p.grad is not None]
        # if not params:
        #    return

        # for p in params:
        self.agc_eps = 1e-3
        p_norm = self.unit_norm(p).clamp_(self.agc_eps)
        g_norm = self.unit_norm(p.grad)
    
        self.agc_clip_val = 1e-3
        max_norm = p_norm * self.agc_clip_val

        clipped_grad = p.grad * (max_norm / g_norm.clamp(min=1e-6))

        new_grads = torch.where(g_norm > max_norm, clipped_grad, p.grad)
        p.grad.detach().copy_(new_grads)


    def step(self, closure=None):
        assert closure is not None, "Need to pass loss to optimizer.step"
        loss = closure
        self.max_loss = max(self.max_loss, loss.item())
        self.ma_loss = 0.9 * self.ma_loss + 0.1 * loss.item()
        effective_lr = self.ma_loss / self.max_loss
        print(effective_lr)

        #Evaluate averages and grad, update param tensors
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                self.agc(p)
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
                    
                    state['square_average'] = torch.zeros_like(p_data_fp32)

                    #look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['1st_moment'] = state['1st_moment'].type_as(p_data_fp32)
                    state['2nd_moment'] = state['2nd_moment'].type_as(p_data_fp32)
                    state['square_average'] = state['square_average'].type_as(p_data_fp32)

                #begin computations
                moment_1, moment_2 = state['1st_moment'], state['2nd_moment']
                beta1, beta2 = group['betas']

                moment_1.mul(beta1).addcmul_(1-beta1, grad, grad)

                #compute 1st_moment moving avg
                moment_1.mul_(beta1).add_(1 - beta1, grad)
                #compute 2nd_moment moving avg
                moment_2.mul_(beta2).addcmul_(1 - beta2, grad, grad)



                # step_size = 1
                # denom = moment_2.sqrt().add_(1e-9) # 1e-9 is eps
                # p_data_fp32.addcdiv_(-step_size * group['lr'], moment_1, denom)

                square_avg = state['square_average']
                square_avg.mul_(beta1).addcmul_(grad, grad, value=1 - beta1)
                avg = square_avg.sqrt().add(1e-9)
                p_data_fp32.addcdiv_(grad, avg, value=-group['lr'])
                
                # copy into optimizer
                p.data.copy_(p_data_fp32)



        return loss
