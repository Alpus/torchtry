from abc import ABCMeta, abstractmethod

from torchtry import Unsavable
from torchtry.experiment import SaveType
from torchtry.utils.pytorch import CyclicalDataloaderCollection


class DataloadersMixin(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cyclical_train_dataloader = self._make_cyclical(self.train_dataloader)
        if self.val_dataloader is not None:
            self._cyclical_val_dataloader = self._make_cyclical(self.val_dataloader)
        else:
            self._cyclical_val_dataloader = None

    @staticmethod
    def _make_cyclical(dataloader):
        if isinstance(dataloader, SaveType):
            object_, savemode = dataloader.object_savemod()
            return savemode(CyclicalDataloaderCollection(object_))
        else:
            return dataloader

    @property
    @abstractmethod
    def train_dataloader(self):
        pass

    @property
    def val_dataloader(self):
        pass

    def get_train_sample(self):
        return next(self._cyclical_train_dataloader)

    def get_val_sample(self):
        if self._cyclical_val_dataloader is None:
            return None
        else:
            return next(self._cyclical_val_dataloader)
