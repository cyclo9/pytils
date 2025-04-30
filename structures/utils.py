def transpose_list_to_dict(arr):
    """Transposes a list of dicts (all the same) into a single dict of lists"""
    map = {key: [d[key] for d in arr] for key in arr[0]}
    return map
