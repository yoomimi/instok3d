import functools
import importlib
import os
from os import path


class ClassRegistry:
    """Maps a string name to a target class."""

    def __init__(self, group_name):
        self.group_name = group_name
        self._name_class_map = {}

    def register(self, name):
        def decorator_fn(target_cls):
            self._name_class_map[name] = target_cls
            return target_cls

        return decorator_fn

    def get_target_cls(self, name, additional_args=None, **kwargs):
        """Return the class registered under ``name`` with the given arguments bound."""
        if name is None:
            return None

        assert name in self._name_class_map.keys(), f'failed to find "{name}" for {self.group_name}'
        additional_args = dict(additional_args) if additional_args is not None else {}
        for key, value in kwargs.items():
            additional_args[key] = value

        return functools.partial(self._name_class_map[name], **additional_args)

    def import_current_directories(self, current_path, current_package):
        """Import every module next to ``current_path`` so that its classes get registered."""
        current_dir = path.dirname(current_path)
        files = [x for x in os.listdir(current_dir) if x.endswith('.py')]
        for file in files:
            file_path = path.join(current_dir, file)
            if file_path == current_path:
                continue

            name = file.replace('.py', '')
            importlib.import_module(f'{current_package}.{name}')
