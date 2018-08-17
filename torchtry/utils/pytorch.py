from collections import OrderedDict
from typing import Any, List

import torch
from torch import nn

from .structures import Items, UntraversableOrderedDict, apply_collection, flatten


def tensors_to_cuda(object_, cuda_device):
    """Turn all tensors to cuda. Other objects will not be touched.

    Args:
        object_: any object.

    Returns:
        object with tensors turned to cuda.
    """

    def to_cuda_if_tensor(object_):
        if isinstance(object_, torch.Tensor):
            return object_.cuda(cuda_device)

    return apply_collection(to_cuda_if_tensor, object_)


def all_to_parallel_cuda(object_, cuda_devices: List[int]):
    """Turn all nn.Module objects to nn.DataParallel with cuda. Assert if find not nn.Module object.

    Args:
        object_: object that contains only nn.Module objects.

    Returns:
        object: with modules turned to nn.DataParallel with cuda.
    """

    def to_parallel_cuda_if_module(object_):
        if isinstance(object_, nn.Module):
            return nn.DataParallel(object_, cuda_devices).cuda()
        else:
            return object_

    return apply_collection(to_parallel_cuda_if_module, object_)


def all_to_state_dict(object_, assert_no_state=True, unparallel=False):
    def to_state_dict(object_):
        if isinstance(object_, nn.DataParallel) and unparallel is True:
            object_ = object_.module
        if not hasattr(object_, 'state_dict'):
            assert assert_no_state, 'Object has no state dict'
            return object_
        else:
            return UntraversableOrderedDict(object_.state_dict())

    return apply_collection(to_state_dict, object_)


class CyclicalDataloaderCollection:
    def __init__(self, dataloaders):
        self._min_len = min([lambda dataloader: len(dataloader) in dataloaders])
        self._dataloaders_iterators = apply_collection(
            lambda dataloader: Items(dataloader=dataloader, iterator=iter(dataloader)),
            dataloaders,
        )

    def __len__(self):
        return self._min_len

    def __iter__(self):
        return self

    def __next__(self):
        return apply_collection(self.cyclical_next, self._dataloaders_iterators)

    def cyclical_next(self, item):
        try:
            return next(item.iterator)
        except StopIteration:
            item.dataloader = iter(item.dataloader)
            return next(item.iterator)
