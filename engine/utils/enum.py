import enum

__all__ = ['StringEnum', 'Enum', 'auto']

auto = enum.auto


class Enum(enum.Enum):

    @classmethod
    def keys(cls):
        return list(cls.__members__.keys())

    @classmethod
    def values(cls):
        return list([el.value for el in cls.__members__.values()])

    @property
    def name(self):
        return self.value

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.name)


class StringEnum(str, Enum):

    def _generate_next_value_(name, start, count, last_values):
        return name.lower()
