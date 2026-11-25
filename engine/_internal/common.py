def parse_list_string(list_str, parser=None):
    """Parse ``"a,b,c"`` into a list; an int ``n`` becomes ``list(range(n))``."""
    if isinstance(list_str, int):
        return list(range(list_str))
    if not isinstance(list_str, str):
        return list_str

    parsed_list = []
    for x in list_str.split(','):
        x = x.strip()
        x = parser(x) if callable(parser) else x
        parsed_list.append(x)

    return parsed_list
