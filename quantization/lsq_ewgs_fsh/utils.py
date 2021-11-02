import sys
import math
import numpy as  np
import torch
from .quantizer import LSQ as quantizer


def add_hooks(trainer):
    scheduler = KScheduler(trainer)

    trainer.register_hooks(loc='before_train', func=[scheduler.init_k])
    #trainer.register_hooks(loc='after_epoch', func=[scheduler.update_k])

class KScheduler:
    def __init__(self, trainer):
        super(KScheduler).__init__()
        self.epochs = trainer.cfg.epochs
        self.end_epoch = int(self.epochs / 6 * 5)
        self.start_k = trainer.cfg.lsq_ewgs_fsh_k
        self.end_k = 0.

    def init_k(self, trainer):
        for m in trainer.model.modules():
            if isinstance(m, quantizer):
                if m.q_n != 0:
                    m.k.fill_(self.start_k)

    def update_k(self, trainer):
        epoch = trainer.memory['epoch']

        if epoch > self.end_epoch:
            return
        elif epoch == self.end_epoch:
            new_k = 0.
            for m in trainer.model.modules():
                if isinstance(m, quantizer):
                    if m.q_n != 0:
                        m.k.fill_(new_k)
            return
        else:
            return

        if epoch >= self.end_epoch:
            return
        
        new_k = self.cal_k(epoch + 1)
        for m in trainer.model.modules():
            if isinstance(m, quantizer):
                if m.q_n != 0:
                    m.k.fill_(new_k)
    
    def cal_k(self, epoch):
        k = self.start_k * (1 + math.cos(math.pi * epoch / self.end_epoch)) / 2
        return k
