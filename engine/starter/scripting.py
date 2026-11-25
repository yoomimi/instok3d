import sys
from os import path
from typing import Callable
import importlib

from engine.utils import Singleton
from engine.dependency import resolver
from .parser import EntryConfigParser


class ScriptingContext(Singleton):
    entry_fn: Callable = None
    config_dir: str = 'config/'
    default_config_name: str = None


def entry_fn(config_dir: str = 'config/', default_config_name: str = None):
    """Decorator that registers the entry function of a script started through ``run_script``."""
    context = ScriptingContext.instance()

    def decorator_fn(fn):
        if context.entry_fn is not None:
            raise RuntimeError(f'entry_fn is already registered ({context.entry_fn.__name__}, new: {fn.__name__})')

        context.entry_fn = fn
        context.config_dir = config_dir
        context.default_config_name = default_config_name
        return fn

    return decorator_fn


def run_script(*args, config_dir='config/scripts', script_dir='scripts/', **kwargs):
    """Import the script given as the first command-line argument and call its ``@entry_fn`` with its config."""
    if len(sys.argv) < 2:
        raise RuntimeError('No script file specified.')

    file_path = sys.argv[1]
    module_path = file_path[:-3] if file_path.endswith('.py') else file_path
    package_name = module_path.replace('/', '.')
    importlib.import_module(package_name)

    context = ScriptingContext.instance()

    # Default config of scripts/foo.py: config/scripts/foo.yaml
    if context.default_config_name is None:
        rel_path = path.relpath(file_path, script_dir)
        config_name = path.splitext(rel_path)[0]
        config_file = path.join(config_dir, config_name + '.yaml')
        if not path.exists(config_file):
            raise FileNotFoundError(
                f'Auto-derived config not found: {config_file}\n'
                f'  Script: {file_path}\n'
                f'  Either create {config_file} or specify default_config_name in @entry_fn()')
        context.default_config_name = path.relpath(
            path.join(config_dir, config_name), context.config_dir)

    parser = EntryConfigParser.instance()
    cfg = parser.parse(context.config_dir, context.default_config_name, is_script_entry=True)
    parser.handle_overrides(cfg, allow_unspecified_key=True)
    fn = context.entry_fn
    fn = resolver.resolve_all_dependencies(cfg, fn)
    fn(*args, **kwargs)
