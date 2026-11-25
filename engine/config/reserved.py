from engine.utils import StringEnum


class ReservedKeys(StringEnum):
    BASE: str = '_base_'
    TARGET: str = '_target_'
    PROVIDER: str = '_provider_'
    OVERWRITE: str = '_overwrite_'
    MODULE_NAME: str = '_modulename_'
