import torchvision.transforms as transforms


def change_normalize(cfg, dataset):
    if hasattr(dataset, 'normalize'):
        if cfg.dataset not in ds.keys():
            raise Warning(f'There is no change in the normalization of {cfg.dataset} dataset for quantized inference.')
        dataset.normarlize = ds[cfg.dataset]['normalize']


ds = {
    'cifar10': {'normalize': transforms.Normalize(
                                    mean=[125/255, 123/255, 114/255],
                                    std=[0.2023, 0.1994, 0.2010])},
    'cifar100': {'normalize': transforms.Normalize(
                                    mean=[129/255, 124/255, 112/255],
                                    std=[0.2673, 0.2564, 0.2762])},
    'imagenet': {'normalize': transforms.Normalize(
                                    mean=[124/255, 116/255, 104/255],
                                    std=[0.229, 0.224, 0.225])},
}
