from os import path
from typing import Optional

from engine.utils import Singleton

__all__ = ['get_current_run_state', 'to_output_dir']


class RunState(Singleton):
    """Process-wide state of the current run."""

    def __init__(self):
        super().__init__()

        self.job_name = None
        self.output_dir = None
        self.solver_name = None
        self.datamodule_name = None
        self.should_save_outputs = False


def get_current_run_state() -> RunState:
    return RunState.instance()


def to_output_dir(*path_tokens) -> Optional[str]:
    """Join ``path_tokens`` to the output directory of the run (None if the run has none)."""
    current_dir = get_current_run_state().output_dir
    if current_dir is None:
        return None
    return path.join(current_dir, *path_tokens)
