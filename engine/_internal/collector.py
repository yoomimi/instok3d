class ResultCollector:
    """Accumulates the dicts returned by the steps of an evaluator into ``{key: [values]}``."""

    def __init__(self):
        self._results_dict = None

    def add(self, result):
        if result is None:
            return

        assert isinstance(result, dict), 'result should be a dict'
        if len(result) == 0:
            return

        if self._results_dict is None:
            self._results_dict = {key: [] for key in result.keys()}

        for key, value in result.items():
            assert key in self._results_dict, f'unknown key of result: {key}'
            self._results_dict[key].append(value)

    def result(self):
        return self._results_dict
