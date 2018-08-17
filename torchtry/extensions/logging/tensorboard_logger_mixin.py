import numbers
import warnings

import torch
from tensorboard_logger import Logger

from torchtry.experiment import Unsavable


class StepLogger(Logger):
    def log_step(self, step: int, log_dict: dict, logger: Logger,
                 need_scalars: bool = False, need_images: bool = False, need_histograms: bool = False) -> None:
        for key, value in log_dict.items():
            if isinstance(value, torch.Tensor):
                # for tensor scalars
                if value.size() == torch.Size([]):
                    if need_scalars is True:
                        value = self._prepare_tensor_to_log(value)
                        logger.log_value(
                            key, value, step
                        )
                # for tensor images
                elif len(value.size()) == 4 and value.size(1) in (1, 3, 4):
                    if need_images is True:
                        value = self._prepare_tensor_to_log(value)
                        logger.log_images(
                            key, value, step
                        )
                # for tensors scalars for histograms
                elif len(value.size()) == 1:
                    if need_histograms is True:
                        logger.log_histogram(
                            key, value, step
                        )
                else:
                    warnings.warn(
                        '"{}" is not logged (is this tensor contains images, scalar or scalars?)'.format(
                            key
                        )
                    )
            elif isinstance(value, numbers.Number):
                # for python scalars
                if need_scalars is True:
                    logger.log_value(
                        key, value, step
                    )
            elif isinstance(value, list) and \
                    all([isinstance(scalar, numbers.Number) for scalar in value]):
                # for python scalars for histograms
                if need_histograms is True:
                    logger.log_histogram(
                        key, value, step
                    )
            else:
                warnings.warn('"{}" is not logged (this object is not a tensor, number or numbers)'.format(key))

    def _prepare_tensor_to_log(self, value: torch.Tensor) -> torch.Tensor:
        value = value.detach()
        return value.cpu() if self.use_cuda() else value


class TensorboardLoggerMixin:
    def __init__(self, *args, **kwargs):
        super(*args, **kwargs)
        self._train_logger = Unsavable(None)
        self._val_logger = Unsavable(None)

    def get_train_logger(self) -> StepLogger:
        if self._train_logger is None:
            self._train_logger = StepLogger(self.train_log_dir())
        return self._train_logger

    def get_val_logger(self) -> Logger:
        if self._val_logger is None:
            self._val_logger = StepLogger(self.val_log_dir())
        return self._val_logger

    @staticmethod
    def tensor_0_1_to_0_255(tensor):
        return (tensor.clamp(0, 1) * 255).type(torch.uint8)
