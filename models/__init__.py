from .utils import *

avail_archs = ['resnet', 'preactresnet', 'mobilenetv2']

for arch in avail_archs:
    exec(f'from .{arch} import set_model as {arch}')

def set_model(cfg, qnn):
    assert cfg.arch in avail_archs, f'The architecture:{cfg.arch} is unimplemented.'
    _local = locals()
    exec(f'model, image_size = {cfg.arch}(cfg, qnn)', globals(), _local)
    return _local['model'], _local['image_size']
