import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


class EWGS_discretizer(torch.autograd.Function):
    """
    x_in: continuous inputs within the range of [0,1]
    num_levels: number of discrete levels
    scaling_factor: backward scaling factor
    x_out: discretized version of x_in within the range of [0,1]
    """
    @staticmethod
    def forward(ctx, x_in, scaling_factor):
        x_out = torch.round(x_in)
        
        ctx._scaling_factor = scaling_factor
        ctx.save_for_backward(x_in-x_out)
        return x_out
    @staticmethod
    def backward(ctx, g):
        diff = ctx.saved_tensors[0]
        delta = ctx._scaling_factor
        scale = 1 + delta * torch.sign(g) * diff
        return g * scale, None


class LSQ(nn.Module):
    def __init__(self, nbits, q_n, q_p):
        super().__init__()
        self.nbits = Parameter(torch.Tensor(1), requires_grad=False)
        self.nbits.fill_(nbits)

        self.q_n = q_n
        self.q_p = q_p

        self.step = Parameter(torch.Tensor(1))
        self.register_buffer('do_init', torch.zeros(1))

        self.register_buffer('bkwd_scaling_factor', torch.tensor(0.).float())
        self.discretizer = EWGS_discretizer.apply
        self.hook_Qvalues = False
        self.buff = None

    @property
    def step_abs(self):
        return self.step.abs()

    def init_step(self, x, *args, **kwargs):
        self.step.data.copy_(
            2. * x.abs().mean() / math.sqrt(self.q_p)
        )
        self.do_init.fill_(1)

    def forward(self, x):
        if self.training and self.do_init == 0:
            self.init_step(x)
        
        step_grad_scale = 1.0 / ((self.q_p * x.numel()) ** 0.5)
        step_scale = grad_scale(self.step_abs, step_grad_scale)

        x = x / step_scale
        x = torch.clamp(x, self.q_n, self.q_p)
        x = self.discretizer(x, self.bkwd_scaling_factor)

        if self.hook_Qvalues:
            self.buff = x
            self.buff.retain_grad()

        x = x * step_scale
        return x

    def extra_repr(self):
        return 'nbits={}, q_n={}, q_p={}'.format(int(self.nbits[0]), int(self.q_n), int(self.q_p))