
avail_quants = ['fp', 'dorefanet', 'pact',
               'lsq', 'lsq_ewgs', 'lsq_ewgs_fsh']

for quant in avail_quants:
    exec(f'from .{quant} import *')