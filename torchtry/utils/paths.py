import os
import shutil


def mkdir_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def remove_if_exists(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    return folder


def folder_is_empty(folder):
    return len(os.listdir(folder)) == 0


def join_and_create_path(*paths):
    return mkdir_if_not_exists(os.path.join(*paths))
