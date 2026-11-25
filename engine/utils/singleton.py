class Singleton:
    __instance = None

    @classmethod
    def instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance
