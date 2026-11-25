def merge_dict(src, target, copy=True):
    """Update ``src`` (a copy of it by default) with the items of ``target``."""
    if copy:
        src = src.copy()
    for key, value in target.items():
        src[key] = value

    return src
