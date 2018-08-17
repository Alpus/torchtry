from .utils.paths import join_and_create_path


__all__ = ['config', 'set_storage_dir']


class Config:
    def __init__(self):
        self._data_dir = None

    def set_data_dir(self, data_dir):
        self._data_dir = data_dir

    @property
    def torchtry_dir(self):
        assert self._data_dir is not None, \
            'Please, set where to create "torchtry" directory using torchtry.set_storage_dir'
        return join_and_create_path(self._data_dir, 'torchtry_storage')

    @property
    def logs_dir(self):
        return join_and_create_path(self.torchtry_dir, 'logs')

    @property
    def error_logs_dir(self):
        return join_and_create_path(self.logs_dir, 'errors')

    @property
    def checkpoints_dir(self):
        return join_and_create_path(self.torchtry_dir, 'checkpoints')


config = Config()


def set_storage_dir(data_dir):
    config.set_data_dir(data_dir)
