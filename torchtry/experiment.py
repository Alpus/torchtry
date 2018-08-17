import os
import traceback
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any, List, Optional, NoReturn

import torch
from datetime import datetime
from tqdm import tqdm

from .config import config
from .utils.paths import folder_is_empty, remove_if_exists, join_and_create_path
from .utils.pytorch import all_to_parallel_cuda, tensors_to_cuda, load_state_dicts, all_to_state_dict
from .utils.time import time_is_up


__all__ = ['Experiment', 'Unsavable']


class SaveType:
    def __init__(self, object_):
        assert isinstance(object_, SaveType), 'Object save type is already defined'
        self._object = object_

    def object_savemod(self):
        return self._object, self.__class__


class Unsavable(SaveType):
    pass


class Savable(SaveType):
    pass


class Experiment(metaclass=ABCMeta):
    """Base class of Experiment

    This class helps to work with experiment with pytorch models training, save checkpoints and log all data. The main
    purpose of this class is reproducibility of experiments and simplification of routine. Inheritor of this class must
    be configuration of model training process without any additional arguments.

    Arguments:
        cuda_devices: train process will not use cuda if None. If argument contains list of integers train process will
            use corresponding devices. If you want to use tricky GPU usage, write your own implementation of
            parallelizing and cuda usage in inheritor's "setup_model_cuda" and "setup_sample_cuda" methods.

    Usage:
        Inheritor must define methods "get_train_sample" and "train_step". Besides it might define "get_val_sample" and
        "val_step" methods. For more details look on methods description.

        There are a few public methods you can use while training pipeline implementation:
            * current_step
            * cuda_devices
            * use_cuda

        After implementation you can use a few methods to control and reproduce experiments:
            * train
            * clean_all
            * get_saved_steps_list
            * get_last_saved_step
            * load_train_step

        During training class will save train steps and can log data:
        * Class save all instance attributes.
            * If you don't want to save something (e.g. too large attribute that not influence on training
                reproducability) you can mark it Unsavable using corresponding class
                "self.attribute = torchtry.Unsavable(value)".
            * All elements, that has ".state_dict()" method will be saved and loaded using this method even if this
                element contained in dict or list.
            * All models save and load as nn.Module.state_dict() even if they use nn.DataParallel wrapper.
        * To use logging you can inherit logger mixins from torchtry.extensions.

        Cuda:
            * All nn.Modules will be automatically converted to nn.DataParallel(module).cuda(self.cuda_devices)
            * All torch.Tensor objects in samples taken from "get_train_sample" and "get_val_sample" methods will
                be automatically converted to a tensor.cuda(self.cuda_devices[0])
    """

    def __init__(self, cuda_devices: Optional[List[int]] = None) -> None:
        self._savemods = defaultdict(Savable)
        self._cuda_devices = Unsavable(cuda_devices)  # type: List[int]

        self._current_step = 0

        super().__init__()

    # INHERITORS MUST HAVE ATTRIBUTES

    @abstractmethod
    def get_train_sample(self) -> Optional[Any]:
        """Return data sample that will be passed to train_step"""

        return None

    def get_val_sample(self) -> Optional[Any]:
        """Return data sample that will be passed to val_step"""

        return None

    @abstractmethod
    def train_step(self, sample) -> Optional[dict]:
        """Makes train step and can return dict with data to log

        Args:
            sample: data from "get_train_sample" method.

        Returns:
            Key of a dict is a name of a data to log. Values must contain data according to logger. Nothing
            will be logged if method returns None.

        Example:
            def train_step(self, sample):
                self.model.train()
                images, goals = sample

                predictions = self.model(images)

                loss_res = nn.functional.mse_loss(predictions, goals)
                self.optimizer.zero_grad()
                loss_res.backward()
                self.optimizer.step()

                return {
                    'loss': loss_res,
                    'images': images,
                    'predictions': predictions,
                    'goals': goals,
                }
        """

        return None

    def val_step(self, sample) -> Optional[dict]:
        """Similar to method "train_dataloader" """

        return None

    # CUDA SETUP

    def setup_model_cuda(self) -> None:
        """Setup model to use cuda"""

        if self.use_cuda():
            self.__dict__ = all_to_parallel_cuda(self.__dict__, self.cuda_devices())

    def setup_sample_cuda(self, sample):
        """Make all torch.Tensor in sample to use cuda"""

        if self.use_cuda():
            return tensors_to_cuda(sample, self.cuda_devices()[0])
        else:
            return sample

    # EXPERIMENT CONTROLL METHODS

    def train(self, save_frequency: int = 60,
              scalars_log_frequency: int = 0.5, images_log_frequency: int = 2, histograms_log_frequency: int = 2,
              validation_frequency: int = None, finish_step: int = None,
              train_type: str = 'careful') -> None:
        """Implements all train pipeline, save checkpoints and log data if corresponding mixin used.

        Args:
            save_frequency: how often make checkpoints (in minutes).
            scalars_log_frequency: how often log scalars (in minutes).
            images_log_frequency: how often log images (in minutes)
            validation_frequency: how often apply model on validation (in minutes)
            finish_step: on which step stop training. Never in case of None.
            train_type:
                * 'careful' starts only if no working files exists,
                * 'continue' starts from last checkpoint,
                * 'clean' removes all data and start from scratch with a confirmation,
                * 'force_clean' removes all data and start from scratch without a confirmation.
        """

        self._prepare_training(train_type)
        self.setup_model_cuda()

        now_time = datetime.now()
        last_save_time = now_time
        last_scalars_log_time = now_time
        last_images_log_time = now_time
        last_histogram_log_time = now_time
        last_validation_time = now_time
        del now_time

        progress_bar = tqdm(
            total=finish_step,
            initial=self.current_step(),
            desc='Step'
        )

        try:
            while True:
                if finish_step is not None and self.current_step() >= finish_step:
                    break

                train_sample = self.get_train_sample()
                train_sample = self.setup_sample_cuda(train_sample)

                train_log_dict = self.train_step(train_sample)

                # log all
                if self.current_step() > 0:
                    # log train
                    need_log_scalars = time_is_up(last_scalars_log_time, scalars_log_frequency)
                    need_log_images = time_is_up(last_images_log_time, images_log_frequency)
                    need_log_histograms = time_is_up(last_histogram_log_time, histograms_log_frequency)

                    if train_log_dict is not None:
                        train_logger = self.get_train_logger()
                        if train_logger is None:
                            warnings.warn(
                                "Training wasn't log: train_step returned dict but get_train_logger returned None. " +
                                'Use logger mixins from torchtry.extensions.'
                            )
                        else:
                            train_logger.log_step(
                                self.current_step(), train_log_dict, self.get_train_logger(),
                                need_scalars=need_log_scalars,
                                need_images=need_log_images,
                                need_histograms=need_log_histograms,
                            )

                    if need_log_scalars is True:
                        last_scalars_log_time = datetime.now()

                    if need_log_images is True:
                        last_images_log_time = datetime.now()

                    if need_log_histograms is True:
                        last_histogram_log_time = datetime.now()

                # start validation
                if time_is_up(last_validation_time, validation_frequency) is True:
                    val_sample = self.get_val_sample()
                    if val_sample is not None:
                        val_sample = self.setup_sample_cuda(val_sample)
                        val_log_dict = self.val_step(val_sample)

                        val_logger = self.get_val_logger()

                        # log validation
                        if val_logger is None:
                            warnings.warn(
                                "Validation wasn't log: val_step returned dict but get_val_logger returned None. " +
                                'Use logger mixins from torchtry.extensions.'
                            )
                        else:
                            val_logger.log_step(
                                self.current_step(), val_log_dict, val_logger,
                                need_scalars=True, need_images=True,
                            )
                    else:
                        warnings.warn(
                            "Validation wasn't done: validation_frequency is not None but val sample equals None."
                        )

                    last_validation_time = datetime.now()

                if time_is_up(last_save_time, save_frequency) is True:
                    self._save_current_train_step()
                    last_save_time = datetime.now()

                self._current_step += 1
                progress_bar.update()
        except KeyboardInterrupt:
            print('Stopped.')
        except Exception:
            traceback_text = traceback.format_exc()
            print(traceback_text)
            self._log_error(traceback_text)
        finally:
            self._save_current_train_step()
            progress_bar.close()

    def clean_all(self, force: bool = False) -> None:
        """Cleans all data related with experiment"""

        if force is False:
            print(
                'Clean all train data for (Y/N):\nlog_dir: {}\ncheckpoints_dir: {}'.format(
                    self.root_log_dir(),
                    self.checkpoints_dir(),
                )
            )
            answer = input().lower()
        else:
            answer = 'y'
        if answer is 'y':
            remove_if_exists(self.root_log_dir())
            remove_if_exists(self.checkpoints_dir())
        else:
            print('NOT cleaned (negative answer)')

    # TRAINING PROCESS HELPERS

    def current_step(self) -> int:
        return self._current_step

    def cuda_devices(self) -> List[int]:
        return self._cuda_devices

    def use_cuda(self) -> bool:
        return self.cuda_devices() is not None

    def name(self) -> str:
        return self.__class__.__name__

    # PATHS

    def root_log_dir(self) -> str:
        return join_and_create_path(config.logs_dir, self.name())

    def train_log_dir(self) -> str:
        return join_and_create_path(self.root_log_dir(), 'train')

    def val_log_dir(self) -> str:
        return join_and_create_path(self.root_log_dir(), 'val')

    def checkpoints_dir(self) -> str:
        return join_and_create_path(config.checkpoints_dir, self.name())

    # INTERIOR HELPERS

    def _prepare_training(self, train_type: str) -> Optional[NoReturn]:
        """Check workspace, load checkpoint if needed"""

        if train_type is 'careful':
            if not (folder_is_empty(self.root_log_dir()) and folder_is_empty(self.checkpoints_dir())):
                raise AssertionError(
                    'There are previous train files:\nlog_dir: {}\ncheckpoints_dir: {}\n'.format(
                        self.root_log_dir(),
                        self.checkpoints_dir(),
                    )
                )
        elif train_type is 'continue':
            try:
                last_checkpoint_step = self.get_last_saved_step()
                self.load_train_step(last_checkpoint_step)
            except Exception:
                raise AssertionError('Nothing to load')
        elif train_type is 'clean':
            self.clean_all()
        elif train_type is 'force_clean':
            self.clean_all(force=True)
        else:
            raise AssertionError('No such train type (use "careful", "continue", "clean" or "force_clean")"')

    def __setattr__(self, key: str, value) -> None:
        if isinstance(value, SaveType):
            value, self._savemods[key] = value.object_savemod()

        super().__setattr__(key, value)

    # CHECKPOINTS MANAGING

    def get_last_saved_step(self) -> int:
        """Returns last checkpoint step number"""

        return self.get_saved_steps_list()[-1]

    def get_saved_steps_list(self) -> List[int]:
        """Returns list of all checkpoint step numbers"""

        return sorted(map(int, os.listdir(self.checkpoints_dir())))

    # HIDDEN CHECKPOINTS PATHS

    def _get_checkpoint_dir(self, step: int) -> str:
        return join_and_create_path(self.checkpoints_dir(), '{:010}'.format(step))

    @staticmethod
    def _get_checkpoint_experiment_state_path(checkpoint_dir: str, key: str) -> str:
        return os.path.join(checkpoint_dir, key)

    # EXPERIMENT STATE

    def _get_experiment_state(self) -> dict:
        experiment_state = dict()

        for key, value in self.__dict__.items():
            if self._savemods[key] is Unsavable:
                continue
            else:
                experiment_state[key] = all_to_state_dict(value, assert_no_state=False, unparallel=True)

        return experiment_state

    def _set_experiment_state(self, experiment_state: dict) -> None:
        self.__dict__.update(experiment_state)

    # CHECKPOINT SAVING

    def _save_current_train_step(self) -> None:
        checkpoint_dir = self._get_checkpoint_dir(self.current_step())
        self._save_experiment_state(checkpoint_dir)

    def _save_experiment_state(self, checkpoint_dir: str) -> None:
        for key, value in self._get_experiment_state().items():
            torch.save(
                value,
                self._get_checkpoint_experiment_state_path(checkpoint_dir, key)
            )

    # CHECKPOINT LOADING

    def load_train_step(self, step: int) -> None:
        """Loads checkpoint on defined step (need to exists)"""

        checkpoint_dir = self._get_checkpoint_dir(step)
        self._load_experiment_state(checkpoint_dir)

    def _load_experiment_state(self, checkpoint_dir: str) -> None:
        self._set_experiment_state(
            {
                key: torch.load(
                    self._get_checkpoint_experiment_state_path(checkpoint_dir, key)
                )
                for key in os.listdir(checkpoint_dir)
            }
        )

    # LOGGING

    def get_train_logger(self):
        return None

    def get_val_logger(self):
        return None

    def _log_error(self, traceback_text: str) -> None:
        error_path = os.path.join(
            config.error_logs_dir,
            self.name() + '_' + datetime.now().strftime(
                '%Y-%m-%d-%H-%M-%S-%f'
            )
        )
        with open(error_path, 'w') as file:
            file.write(traceback_text)
