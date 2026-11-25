from engine.utils.enum import StringEnum, auto

__all__ = ['Phases']


class Phases(StringEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
