import sys
import numpy as  np
import torch
from .quantizer import LSQ as quantizer
#from .qnn import QuantConv2d as quantizer


def add_hooks(trainer):
    trainer.register_hooks(loc='before_epoch', func=[update_grad_scales])


def update_grad_scales(trainer):
    if trainer.memory['epoch'] % 10 != 0 or trainer.memory['epoch'] == 0:
        return
    
    model = trainer.model
    train_loader = trainer.loaders['train']
    criterion = trainer.criterion
    device = trainer.device

    ## update scales
    scales = []
    for m in model.modules():
        if isinstance(m, quantizer):
            m.hook_Qvalues = True
            scales.append(0)
    
    model.train()
    for num_batches, (images, labels) in enumerate(train_loader):
        if num_batches == 3: # estimate trace using 3 batches
            break
        images = images.to(device)
        labels = labels.to(device)

        # forward with single batch
        model.zero_grad()
        pred = model(images)
        loss = criterion(pred, labels)
        loss.backward(create_graph=True)

        # store quantized values
        Qvalues = []
        orders = []
        i = 0
        for m in model.modules():
            if isinstance(m, quantizer):
                Qvalues.append(m.buff)

                if m.q_n == 0: # activation
                    orders.insert(0, i)
                else: # conv and fc
                    orders.append(i)
                i += 1

        # update the scaling factor for activations
        for i in orders:
            param = Qvalues[i]
            grad = Qvalues[i].grad

            trace_hess = np.mean(trace(model, [param], [grad], device))
            avg_trace_hess = trace_hess / param.view(-1).size()[0] # avg trace of hessian
            scales[i] += (avg_trace_hess / (grad.std().cpu().item()*3.0))        

    i = 0
    for m in model.modules():
        if isinstance(m, quantizer):
            scales[i] /= num_batches
            scales[i] = np.clip(scales[i], 0, np.inf)
            
            m.bkwd_scaling_factor.data.fill_(scales[i])
            m.hook_Qvalues = False
            i += 1


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv

def trace(model, params, grads, device, maxIter=50, tol=1e-3):
    """
    compute the trace of hessian using Hutchinson's method
    maxIter: maximum iterations used to compute trace
    tol: the relative tolerance
    """

    trace_vhv = []
    trace = 0.

    for i in range(maxIter):
        model.zero_grad()
        v = [
            torch.randint_like(p, high=2, device=device)
            for p in params
        ]
        # generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        
        Hv = hessian_vector_product(grads, params, v)
        trace_vhv.append(group_product(Hv, v).cpu().item())
        if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
            return trace_vhv
        else:
            trace = np.mean(trace_vhv)

    return trace_vhv