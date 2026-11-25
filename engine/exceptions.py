class EngineException(Exception):
    """Base class of the engine errors."""


class ConfigNotFound(EngineException):

    def __init__(self, config_path):
        super().__init__(f'Config file not found: {config_path}')


class ModuleImportException(EngineException):

    def __init__(self, module_name, e):
        if isinstance(e, str):
            message = f'({module_name}) {e}'
        else:
            error_name = e.__class__.__name__ if e is not None else ''
            message = f'({module_name}) {error_name}: {e.args[0]}'

        super().__init__(message)


class TargetLoadException(EngineException):

    def __init__(self, target_name, message=None):
        if message is None:
            message = f'Failed to load target: {target_name}'
        super().__init__(message)


class UnknownKeyException(EngineException):

    def __init__(self, target, unknown_keys):
        super().__init__(f'({target.__name__}) Unknown keys: {", ".join(unknown_keys)}')


class InstantiateException(EngineException):

    def __init__(self, e, target):
        target_name = target.__name__
        error_name = e.__class__.__name__
        error_message = e.args[0] if len(e.args) > 0 else ''

        super().__init__(f'({target_name}) {error_name}: {error_message}')


class DependencyResolveException(EngineException):

    def __init__(self, msg):
        super().__init__(f'(DependencyResolveException) {msg}')
