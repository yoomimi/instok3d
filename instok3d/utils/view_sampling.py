from typing import Tuple


def build_uniform_frame_ids(start: int, stride: int, num_views: int) -> Tuple[int, ...]:
    """Frame ids with uniform spacing from a start index."""
    return tuple(int(start) + i * int(stride) for i in range(int(num_views)))
