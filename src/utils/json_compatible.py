import numpy as np

def json_compatible(obj):
    """
    递归将 numpy 数组 / 标量 / Pandas 类型转换为 JSON 可序列化的 Python 类型。
    """
    if isinstance(obj, dict):
        return {json_compatible(k): json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [json_compatible(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return json_compatible(list(obj))
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif obj is None:
        return None
    else:
        return obj
